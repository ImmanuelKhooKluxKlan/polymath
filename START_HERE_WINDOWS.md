# Start Polymath Musician on Windows

## 1. Install clean dependencies

The delivered folder intentionally excludes `node_modules` so Windows installs the correct native packages for your computer.

```powershell
cd D:\path\to\Polymath_Musician
npm install
npm --prefix server install
```

## 2. Configure environment files

```powershell
Copy-Item .env.example .env
Copy-Item server\.env.example server\.env
```

Open `server\.env` and add your real PayPal, YouTube, and OpenAI values. Keep `OPENAI_API_KEY` only in this backend file. Use a PayPal plan configured for **USD 19.99/month**. Keep `WELCOME_MCOINS=0` for a real launch.

## 3. Start the backend

```powershell
npm run server
```

## 4. Start the frontend in a second PowerShell window

```powershell
cd D:\path\to\Polymath_Musician
npm run dev
```

## 5. Release validation

```powershell
npm run lint
npm run build
node --check server\server.js
```

PDF translation stays unavailable without charging users until `OPENAI_API_KEY` is configured in `server/.env`.
