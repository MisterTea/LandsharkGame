exports.up = async function (knex) {
  await knex.schema.createTable("users", (table) => {
    table.increments("id");
    table.string("email").notNullable();
  });

  await knex("users").insert([{ email: "test@test.com" }]);
};

exports.down = async function (knex) {
  await knex.raw("DROP TABLE users CASCADE");
};
