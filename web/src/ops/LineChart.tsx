import { type MouseEvent, useState } from "react";

export interface Point {
  key: string;
  label: string; // x-axis / tooltip label
  value: number;
  detail?: string; // extra tooltip line
}

interface Props {
  points: Point[];
  title: string;
  format: (value: number) => string;
  /** Upper end of the y axis; defaults to the largest value. The axis always starts at 0. */
  yMax?: number;
  height?: number;
}

const W = 600;
const PAD = { top: 12, right: 12, bottom: 24, left: 44 };

/**
 * One series as a 2 px line with markers, a recessive grid, and a hover crosshair with a
 * tooltip. Single series, so no legend: the title names it. Text uses text colours, never
 * the series colour. Every chart on the dashboard also has a table with the same numbers.
 */
export function LineChart({ points, title, format, yMax, height = 160 }: Props) {
  const [hover, setHover] = useState<number>();
  if (points.length === 0) return null;

  const top = Math.max(yMax ?? 0, ...points.map((p) => p.value)) || 1;
  const innerW = W - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const y = (v: number) => PAD.top + innerH - (v / top) * innerH;
  const ticks = [0, top / 2, top];
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");

  const onMove = (event: MouseEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    setHover(Math.round(Math.min(1, Math.max(0, fraction)) * (points.length - 1)));
  };
  const active = hover === undefined ? undefined : points[hover];

  return (
    <figure className="relative">
      <figcaption className="mb-1 text-sm font-medium text-slate-700">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full" role="img" aria-label={`${title}: ${points.length} points, latest ${format(points.at(-1)!.value)}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="stroke-slate-200" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" className="fill-slate-500 text-[10px]">
              {format(t)}
            </text>
          </g>
        ))}
        <text x={PAD.left} y={height - 6} className="fill-slate-500 text-[10px]">{points[0]!.label}</text>
        {points.length > 1 && (
          <text x={W - PAD.right} y={height - 6} textAnchor="end" className="fill-slate-500 text-[10px]">{points.at(-1)!.label}</text>
        )}
        {active && <line x1={x(hover!)} x2={x(hover!)} y1={PAD.top} y2={PAD.top + innerH} className="stroke-slate-400" strokeWidth={1} strokeDasharray="3 3" />}
        <path d={path} fill="none" className="stroke-sky-600" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <circle key={p.key} cx={x(i)} cy={y(p.value)} r={hover === i ? 5 : 4} className="fill-sky-600 stroke-white" strokeWidth={2} />
        ))}
        {/* hit area: the whole plot, larger than the markers */}
        <rect
          data-testid="chart-hit-area"
          x={PAD.left}
          y={PAD.top}
          width={innerW}
          height={innerH}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(undefined)}
        />
      </svg>
      {active && (
        <div
          role="tooltip"
          className="pointer-events-none absolute top-6 rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-700 shadow"
          style={{ left: `${(x(hover!) / W) * 100}%`, transform: hover! > points.length / 2 ? "translateX(-105%)" : "translateX(5%)" }}
        >
          <div className="font-semibold text-slate-900">{format(active.value)}</div>
          <div>{active.label}</div>
          {active.detail && <div className="text-slate-500">{active.detail}</div>}
        </div>
      )}
    </figure>
  );
}
