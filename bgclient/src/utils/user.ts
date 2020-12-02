import { useEffect, useState } from "react";

var __user = null;

export async function fetchUser(cookie = "") {
  if (typeof window !== "undefined" && __user) {
    return __user;
  }

  const res = await fetch(
    "/api/AuthProfile",
    cookie
      ? {
          headers: {
            cookie,
          },
        }
      : {}
  );

  if (!res.ok) {
    __user = null;
    return null;
  }

  const json = await res.json();
  if (typeof window !== "undefined") {
    __user = json;
  }

  {
    console.log("FETCHING ME");
    const res = await fetch(
      "/api/Me",
      cookie
        ? {
            headers: {
              cookie,
            },
          }
        : {}
    );

    if (!res.ok) {
      console.log("NOT OK");
      __user = null;
      return null;
    }

    const json = await res.json();
    console.log(json);
    // Verify that user exists
    if (json.returning) {
      console.log("OLD USER");
    } else {
      console.log("NEW USER");
    }
  }

  return json;
}

export function useFetchUser({ required } = { required: false }) {
  const [loading, setLoading] = useState(
    () => !(typeof window !== "undefined" && __user)
  );
  const [user, setUser] = useState(() => {
    if (typeof window === "undefined") {
      return null;
    }

    return __user || null;
  });

  useEffect(
    () => {
      if (!loading && user) {
        return;
      }
      setLoading(true);
      let isMounted = true;

      fetchUser().then((user) => {
        // Only set the user if the component is still mounted
        if (isMounted) {
          // When the user is not logged in but login is required
          if (required && !user) {
            window.location.href = "/api/login";
            return;
          }
          setUser(user);
          setLoading(false);
        }
      });

      return () => {
        isMounted = false;
      };
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  return { user, loading };
}
