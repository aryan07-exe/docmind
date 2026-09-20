const raw = process.env.REACT_APP_API_URL;
export const API = raw === undefined ? "http://localhost:8000" : raw.replace(/\/$/, "");
