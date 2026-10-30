import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, GitBranch, Loader2, AlertTriangle, ChevronLeft, ChevronRight } from 'lucide-react';
import { API_BASE } from '../config';

/**
 * Fetches one tree of the clinical XGBoost ensemble (GET /clinical/tree/{index})
 * and draws it as an SVG. The index selector covers 0 to n_estimators-1.
 *
 * Props:
 *   open: bool
 *   onClose: () => void
 */

// Strip the ColumnTransformer's "numeric__"/"binary__"/"categorical__"
// prefix for a cleaner label (e.g. "numeric__AFP_ng_mL" -> "AFP_ng_mL").
function cleanFeatureName(name) {
  if (!name) return name;
  const idx = name.indexOf('__');
  return idx >= 0 ? name.slice(idx + 2) : name;
}

// Simple layout: y = depth, x = leaf order, so subtrees never overlap.
function layoutTree(root) {
  let nextLeafX = 0;

  function findChild(node, nodeId) {
    return (node.children || []).find((c) => c.nodeid === nodeId) || null;
  }

  function assign(node, depth) {
    if (node.leaf !== undefined) {
      const x = nextLeafX++;
      return { node, depth, x, isLeaf: true };
    }
    const yesChild = findChild(node, node.yes);
    const noChild = findChild(node, node.no);
    const yesLaid = yesChild ? assign(yesChild, depth + 1) : null;
    const noLaid = noChild ? assign(noChild, depth + 1) : null;
    const xs = [yesLaid, noLaid].filter(Boolean).map((c) => c.x);
    const x = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : nextLeafX++;
    return { node, depth, x, isLeaf: false, yesLaid, noLaid };
  }

  return assign(root, 0);
}

function countLeaves(node) {
  if (node.leaf !== undefined) return 1;
  const children = node.children || [];
  return children.reduce((sum, c) => sum + countLeaves(c), 0) || 1;
}

function maxDepth(node, depth = 0) {
  const children = node.children || [];
  if (!children.length) return depth;
  return Math.max(...children.map((c) => maxDepth(c, depth + 1)));
}

const NODE_W = 168;
const NODE_H = 52;
const X_SPACING = 190;
const Y_SPACING = 96;
const PAD = 40;

function TreeNode({ laid, onHover }) {
  const cx = laid.x * X_SPACING + PAD;
  const cy = laid.depth * Y_SPACING + PAD;
  const isLeaf = laid.isLeaf;

  return (
    <g>
      {laid.yesLaid && (
        <Edge from={{ x: cx, y: cy }} to={{ x: laid.yesLaid.x * X_SPACING + PAD, y: laid.yesLaid.depth * Y_SPACING + PAD }} label="yes" positive />
      )}
      {laid.noLaid && (
        <Edge from={{ x: cx, y: cy }} to={{ x: laid.noLaid.x * X_SPACING + PAD, y: laid.noLaid.depth * Y_SPACING + PAD }} label="no" positive={false} />
      )}

      <foreignObject x={cx - NODE_W / 2} y={cy - NODE_H / 2} width={NODE_W} height={NODE_H}>
        <div
          onMouseEnter={() => onHover?.(laid.node)}
          onMouseLeave={() => onHover?.(null)}
          className={`w-full h-full rounded-lg border flex flex-col items-center justify-center text-center px-2 py-1 cursor-default
            ${isLeaf
              ? 'bg-cyan-500/10 border-cyan-500/30'
              : 'bg-white/[0.03] border-white/[0.1]'}`}
        >
          {isLeaf ? (
            <>
              <span className="text-[9px] text-slate-500 uppercase tracking-wider">Leaf</span>
              <span className={`text-[13px] font-mono font-bold ${laid.node.leaf >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                {laid.node.leaf.toFixed(3)}
              </span>
            </>
          ) : (
            <>
              <span className="text-[10.5px] font-semibold text-slate-200 truncate max-w-full" title={cleanFeatureName(laid.node.split)}>
                {cleanFeatureName(laid.node.split)}
              </span>
              <span className="text-[9.5px] font-mono text-slate-500">
                &lt; {Number(laid.node.split_condition).toFixed(3)}
              </span>
            </>
          )}
        </div>
      </foreignObject>

      {laid.yesLaid && <TreeNode laid={laid.yesLaid} onHover={onHover} />}
      {laid.noLaid && <TreeNode laid={laid.noLaid} onHover={onHover} />}
    </g>
  );
}

function Edge({ from, to, label, positive }) {
  const midY = (from.y + to.y) / 2;
  return (
    <g>
      <path
        d={`M ${from.x} ${from.y + NODE_H / 2} C ${from.x} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y - NODE_H / 2}`}
        fill="none"
        stroke={positive ? 'rgba(52,211,153,0.35)' : 'rgba(248,113,113,0.35)'}
        strokeWidth={1.5}
      />
      <text
        x={(from.x + to.x) / 2}
        y={midY}
        textAnchor="middle"
        className="fill-slate-500"
        fontSize={9}
        fontFamily="monospace"
      >
        {label}
      </text>
    </g>
  );
}

export default function DecisionTreeModal({ open, onClose }) {
  const [treeIndex, setTreeIndex] = useState(0);
  const [nEstimators, setNEstimators] = useState(null);
  const [treeData, setTreeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const fetchTree = useCallback(async (index) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/clinical/tree/${index}`);
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Failed to load tree (${res.status})`);
      }
      const data = await res.json();
      setTreeData(data.tree);
      setNEstimators(data.n_estimators);
    } catch (err) {
      setError(err.message);
      setTreeData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (open) fetchTree(treeIndex);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose?.(); };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  const goToTree = (index) => {
    if (nEstimators == null || index < 0 || index >= nEstimators) return;
    setTreeIndex(index);
    fetchTree(index);
  };

  const laidOut = useMemo(() => (treeData ? layoutTree(treeData) : null), [treeData]);
  const svgWidth = useMemo(() => (treeData ? countLeaves(treeData) * X_SPACING + PAD * 2 : 400), [treeData]);
  const svgHeight = useMemo(() => (treeData ? (maxDepth(treeData) + 1) * Y_SPACING + PAD * 2 : 300), [treeData]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="tree-backdrop"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            onClick={onClose}
            className="fixed inset-0 z-[90] bg-navy-950/70 backdrop-blur-md"
          />
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6 pointer-events-none">
            <motion.div
              key="tree-modal"
              initial={{ opacity: 0, scale: 0.96, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 10 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
              className="glass-panel glow-border-cyan flex flex-col overflow-hidden relative
                w-full max-w-[1100px] h-[85vh] pointer-events-auto shadow-2xl shadow-black/60 ring-1 ring-white/10"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06] shrink-0">
                <div className="flex items-center gap-2.5">
                  <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/15">
                    <GitBranch size={14} className="text-cyan-400" />
                  </div>
                  <h3 className="text-[13px] font-semibold text-slate-200 tracking-wide uppercase">
                    Decision Tree Viewer
                  </h3>
                </div>
                <button onClick={onClose} className="p-1.5 rounded-md hover:bg-white/5 text-slate-500 hover:text-slate-300 transition-colors">
                  <X size={15} />
                </button>
              </div>

              {/* Tree index selector */}
              <div className="flex items-center gap-3 px-4 py-2.5 border-b border-white/[0.06] shrink-0">
                <button
                  onClick={() => goToTree(treeIndex - 1)}
                  disabled={treeIndex <= 0 || loading}
                  className="p-1.5 rounded-md bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:text-cyan-300 disabled:opacity-30 transition-colors"
                >
                  <ChevronLeft size={14} />
                </button>

                <input
                  type="range"
                  min={0}
                  max={nEstimators != null ? nEstimators - 1 : 0}
                  value={treeIndex}
                  onChange={(e) => goToTree(Number(e.target.value))}
                  disabled={nEstimators == null}
                  className="flex-1 accent-cyan-500"
                />

                <button
                  onClick={() => goToTree(treeIndex + 1)}
                  disabled={nEstimators == null || treeIndex >= nEstimators - 1 || loading}
                  className="p-1.5 rounded-md bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:text-cyan-300 disabled:opacity-30 transition-colors"
                >
                  <ChevronRight size={14} />
                </button>

                <span className="text-[11px] font-mono text-slate-400 whitespace-nowrap">
                  Tree {treeIndex}{nEstimators != null ? ` / ${nEstimators - 1}` : ''}
                </span>
              </div>

              {/* Tree canvas */}
              <div className="flex-1 overflow-auto p-6 bg-black/20">
                {loading && (
                  <div className="flex items-center justify-center gap-2 h-full text-cyan-300">
                    <Loader2 size={16} className="animate-spin" /> Loading tree…
                  </div>
                )}
                {error && !loading && (
                  <div className="flex items-center justify-center gap-2 h-full text-red-400 text-sm">
                    <AlertTriangle size={16} /> {error}
                  </div>
                )}
                {!loading && !error && laidOut && (
                  <svg width={svgWidth} height={svgHeight} className="mx-auto">
                    <TreeNode laid={laidOut} onHover={setHoveredNode} />
                  </svg>
                )}
              </div>

              {/* Footer: hovered-node detail / legend */}
              <div className="px-4 py-2.5 border-t border-white/[0.06] shrink-0 flex items-center justify-between text-[10.5px] text-slate-500">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1.5"><span className="w-2 h-0.5 bg-emerald-400/60 inline-block" /> yes (condition true)</span>
                  <span className="flex items-center gap-1.5"><span className="w-2 h-0.5 bg-red-400/60 inline-block" /> no</span>
                </div>
                {hoveredNode && (
                  <span className="font-mono text-slate-400">
                    {hoveredNode.leaf !== undefined
                      ? `leaf value: ${hoveredNode.leaf.toFixed(4)}`
                      : `${cleanFeatureName(hoveredNode.split)} < ${Number(hoveredNode.split_condition).toFixed(4)}`}
                  </span>
                )}
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
