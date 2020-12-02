import auth0 from '../../utils/auth0';

interface User {
  id: number;
  authId: string;
}

export default async function authProfile(req, res) {
  try {
    await auth0.handleProfile(req, res);
  } catch (error) {
    console.error(error)
    res.status(error.status || 500).end(error.message)
  }
}
