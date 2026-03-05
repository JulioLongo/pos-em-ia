import express from 'express';
import pg from 'pg';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const { Pool } = pg;
const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
// Aumenta o limite para acomodar os vetores serializados dos filmes e usuários
app.use(express.json({ limit: '10mb' }));

// Serve os arquivos estáticos do app de filmes como raiz do servidor
app.use('/', express.static(join(__dirname, 'movies')));

// Conexão com o PostgreSQL (pgvector)
const pool = new Pool({
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME || 'vectors',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
});

/**
 * Cria (ou recria) as tabelas de vetores no banco com a dimensão correta.
 * Recria as tabelas a cada novo treinamento para garantir consistência de dimensão.
 */
async function initDB(dimensions) {
    // Valida que dimensions é um inteiro positivo para evitar SQL injection
    const dim = parseInt(dimensions, 10);
    if (!Number.isInteger(dim) || dim <= 0) {
        throw new Error(`Dimensão inválida: ${dimensions}`);
    }

    await pool.query('CREATE EXTENSION IF NOT EXISTS vector');

    // Recria as tabelas com a dimensão do vetor do modelo atual
    await pool.query('DROP TABLE IF EXISTS movie_vectors');
    await pool.query('DROP TABLE IF EXISTS user_vectors');

    await pool.query(`
        CREATE TABLE movie_vectors (
            movie_id  INTEGER PRIMARY KEY,
            name      TEXT,
            meta      JSONB,
            vector    vector(${dim})
        )
    `);

    await pool.query(`
        CREATE TABLE user_vectors (
            user_id INTEGER PRIMARY KEY,
            name    TEXT,
            age     INTEGER,
            watched JSONB,
            vector  vector(${dim})
        )
    `);
}

/**
 * POST /api/vectors/save
 * Recebe os vetores de filmes e usuários calculados pelo worker e os persiste no pgvector.
 * Body: { movieVectors, userVectors, dimensions }
 */
app.post('/api/vectors/save', async (req, res) => {
    const { movieVectors, userVectors, dimensions } = req.body;

    try {
        await initDB(dimensions);

        for (const mv of movieVectors) {
            await pool.query(
                `INSERT INTO movie_vectors (movie_id, name, meta, vector)
                 VALUES ($1, $2, $3, $4::vector)`,
                [mv.movie_id, mv.name, JSON.stringify(mv.meta), `[${mv.vector.join(',')}]`]
            );
        }

        for (const uv of userVectors) {
            await pool.query(
                `INSERT INTO user_vectors (user_id, name, age, watched, vector)
                 VALUES ($1, $2, $3, $4, $5::vector)`,
                [uv.user_id, uv.name, uv.age, JSON.stringify(uv.watched), `[${uv.vector.join(',')}]`]
            );
        }

        console.log(`Vetores salvos: ${movieVectors.length} filmes, ${userVectors.length} usuários`);
        res.json({ success: true });
    } catch (err) {
        console.error('Erro ao salvar vetores:', err.message);
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /api/similar-users
 * Busca os N usuários mais próximos ao vetor informado usando distância L2 (<->) do pgvector.
 * Body: { vector: number[], limit: number }
 * Retorna: { similarUsers: [{ user_id, name, age, watched }] }
 */
app.post('/api/similar-users', async (req, res) => {
    const { vector, limit = 100 } = req.body;

    // Valida o vetor recebido
    if (!Array.isArray(vector) || vector.length === 0) {
        return res.status(400).json({ error: 'Vetor inválido' });
    }

    try {
        const vectorStr = `[${vector.join(',')}]`;
        const result = await pool.query(
            `SELECT user_id, name, age, watched
             FROM user_vectors
             ORDER BY vector <-> $1::vector
             LIMIT $2`,
            [vectorStr, limit]
        );

        res.json({ similarUsers: result.rows });
    } catch (err) {
        console.error('Erro ao buscar usuários similares:', err.message);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`\n🎬 Movies Recommendation App rodando em: http://localhost:${PORT}`);
    console.log(`📦 Certifique-se de que o Docker está rodando: docker compose up -d\n`);
});
