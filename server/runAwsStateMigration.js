const { loadAwsEnvironment } = require('./startAwsRuntime');

async function run() {
  await loadAwsEnvironment();
  const { main } = require('./migrateStateToPostgres');
  await main();
}

if (require.main === module) {
  run().catch((error) => {
    console.error('Polymath AWS state migration failed:', error.message || error);
    process.exitCode = 1;
  });
}

module.exports = { run };
