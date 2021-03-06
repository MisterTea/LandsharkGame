import Layout from "../components/Layout";
import { useFetchUser } from "../utils/user";

function Home() {
  const { user, loading } = useFetchUser();

  const build_id = process.env.NEXT_PUBLIC_CONFIG_BUILD_ID;

  return (
    <Layout user={user} loading={loading}>
      <h1>Next.js and Auth0 Example</h1>

      {loading && <p>Loading login info...</p>}

      {!loading && !user && (
        <>
          <p>
            To test the login click in <i>Login</i>
          </p>
          <p>
            Once you have logged in you should be able to click in{" "}
            <i>Profile</i> and <i>Logout</i>
          </p>
        </>
      )}

      {user && (
        <>
          <h4>Rendered user info on the client</h4>
          <img src={user.picture} alt="user picture" />
          <p>nickname: {user.nickname}</p>
          <p>name: {user.name}</p>
          <p>BUILD ID: {build_id}</p>
        </>
      )}
    </Layout>
  );
}

export default Home;
