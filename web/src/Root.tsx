import { lazy, Suspense } from "react";
import App from "./App";

// The ops dashboard is a separate chunk: users of the Q&A page never download it.
const Dashboard = lazy(() => import("./ops/Dashboard"));

export function Root({ path }: { path: string }) {
  if (path.replace(/\/+$/, "") !== "/ops") return <App />;
  return (
    <Suspense fallback={<p className="p-4 text-sm text-slate-500">Loading…</p>}>
      <Dashboard />
    </Suspense>
  );
}
