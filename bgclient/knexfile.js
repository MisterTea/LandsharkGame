const { loadEnvConfig } = require("@next/env");

const dev = process.env.NODE_ENV !== "production";
const { PG_URI } = loadEnvConfig("./", dev).combinedEnv;

if (dev) {
  module.exports = {
    client: "sqlite3",
    connection: {
      filename: "./db/knex.sqlite3",
    },
    migrations: {
      directory: "./src/knex/migrations",
    },
    seeds: {
      directory: "./src/knex/seeds",
    },
  };
} else {
  module.exports = {
    client: "pg",
    connection: PG_URI,
    migrations: {
      directory: "./src/knex/migrations",
    },
    seeds: {
      directory: "./src/knex/seeds",
    },
  };
}
