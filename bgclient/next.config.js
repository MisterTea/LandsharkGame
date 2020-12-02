const { PHASE_DEVELOPMENT_SERVER } = require("next/constants");

module.exports = (phase, { defaultConfig }) => {
  var options;
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    options = {};
  } else {
    options = {};
  }

  retval = {
    ...options,
    ...{
      webpack: (config, options) => {
        config.plugins.push(
          new options.webpack.DefinePlugin({
            "process.env.NEXT_PUBLIC_CONFIG_BUILD_ID": JSON.stringify(
              options.buildId
            ),
          })
        );
        return config;
      },
    },
  };
  console.log(retval);
  return retval;
};
