import { getKnex } from "../../knex";
import auth0 from "../../utils/auth0";

interface User {
  id: number;
  authId: string;
}

export default async function me(req, res) {
  try {
    const session = await auth0.getSession(req);
    console.log("GOT SESSION");
    console.log(session);
    const knex = getKnex();
    const user = await knex<User>("users")
      .where("email", session.user.email)
      .first();
    console.log(user);
    if (user) {
      res.json({ returning: true });
    } else {
      var Analytics = require("analytics-node");
      var analytics = new Analytics("AC29Exso0mAGGsiHVkDPx7ramnCRZTbk");
      analytics.identify({
        userId: session.user.email,
        traits: {
          name: session.user.name,
          email: session.user.email,
        },
      });
      analytics.track({
        userId: session.user.email,
        event: "Signed Up",
        properties: {
          foo: "bar",
        },
      });
      res.json({ returning: false });
    }
  } catch (error) {
    console.error(error);
    res.status(error.status || 500).end(error.message);
  }
}
