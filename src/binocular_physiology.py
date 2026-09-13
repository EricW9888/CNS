"""Neural-only binocular physiology model with separate interaction operators.

Chemical anatomy is compiled once from MaleCNS. Gap pairs are supplied
explicitly by the frozen literature specification. No stimulus label enters
the numerical update; only a time-by-neuron external drive does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .malecns_io import ConnectomeGraph
from .numerics import SparseDiffusiveCoupling, effective_euler_transition, spectral_radius
from .sparse_backend import StateRecorder, compile_projection

POPULATIONS = ("HS", "H2", "bIPS", "uLPTCrn", "H2rn", "DNp15")
CONTROLS = ("full", "no_electrical", "no_bIPS_chemical_output", "no_bIPS_GABA_input", "no_DNp15_GABA_input", "no_chemical_feedback", "feedforward_chemical_only")
INTERMEDIATES = {"bIPS", "uLPTCrn", "H2rn"}


@dataclass(frozen=True)
class PhysiologyParameters:
    dt_ms: float = .5
    tau_ms: float = 20.
    chemical_gain: float = .5
    gap_conductance: float = .1
    input_amplitude: float = 1.
    onset_ms: float = 2000.
    offset_ms: float = 4000.
    duration_ms: float = 6000.

    def __post_init__(self):
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("parameters must be finite")
        if self.dt_ms <= 0 or self.tau_ms <= 0 or self.duration_ms <= 0:
            raise ValueError("dt, tau and duration must be positive")
        if min(self.chemical_gain, self.gap_conductance, self.input_amplitude) < 0:
            raise ValueError("gains/conductance/amplitude cannot be negative")
        if not 0 <= self.onset_ms < self.offset_ms <= self.duration_ms:
            raise ValueError("invalid stimulus time interval")
        if not np.isclose(self.duration_ms/self.dt_ms, round(self.duration_ms/self.dt_ms)):
            raise ValueError("duration must be an integer number of steps")


@dataclass(frozen=True)
class BinocularNetwork:
    nodes: pd.DataFrame
    chemical: sparse.csr_matrix
    electrical: SparseDiffusiveCoupling
    parameters: PhysiologyParameters
    control: str
    removed_chemical_edges: int

    @classmethod
    def compile(cls, graph: ConnectomeGraph, electrical_pairs: list[dict],
                parameters: PhysiologyParameters, control: str = "full") -> "BinocularNetwork":
        if control not in CONTROLS:
            raise ValueError("unknown mechanistic control")
        nodes = graph.nodes.sort_values("bodyId").reset_index(drop=True).copy()
        if nodes.bodyId.duplicated().any() or not nodes.stage.isin(POPULATIONS).all():
            raise ValueError("invalid body/population axes")
        if not nodes.somaSide.isin(["L", "R"]).all():
            raise ValueError("unresolved anatomical side")
        expected = np.where(nodes.stage.isin(INTERMEDIATES), "gaba", "acetylcholine")
        if not np.equal(nodes.consensus_nt.to_numpy(), expected).all():
            raise ValueError("unresolved transmitter/sign")
        ids = nodes.bodyId.to_numpy(dtype=np.int64)
        metadata = nodes.set_index("bodyId")
        edges = graph.edges.copy()
        if edges.duplicated(["pre_body", "post_body"]).any():
            raise ValueError("duplicate aggregate chemical edges")
        if not edges.pre_body.isin(ids).all() or not edges.post_body.isin(ids).all():
            raise ValueError("chemical endpoints outside node axes")
        counts = edges.synapse_count.to_numpy(dtype=float)
        if not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError("chemical counts must be finite and positive")
        edges["physiology_sign"] = edges.pre_body.map(metadata.consensus_nt.map({"gaba": -1., "acetylcholine": 1.}))
        # Compile/normalize the FULL operator first. Ablation never changes a
        # survivor's gain or denominator, unlike post-ablation normalization.
        projection = compile_projection(ConnectomeGraph(nodes, edges), source_ids=ids, target_ids=ids,
                                        normalization="absolute", sign_column="physiology_sign")
        chemical = projection.matrix * parameters.chemical_gain
        pre_pop = edges.pre_body.map(metadata.stage)
        post_pop = edges.post_body.map(metadata.stage)
        remove = np.zeros(len(edges), dtype=bool)
        if control == "no_bIPS_chemical_output":
            remove = pre_pop.eq("bIPS").to_numpy()
        elif control == "no_bIPS_GABA_input":
            remove = (pre_pop.isin(INTERMEDIATES) & post_pop.eq("bIPS")).to_numpy()
        elif control == "no_DNp15_GABA_input":
            remove = (pre_pop.isin(INTERMEDIATES) & post_pop.eq("DNp15")).to_numpy()
        elif control == "no_chemical_feedback":
            # Remove intermediate -> input and intermediate -> intermediate
            # chemical feedback. Preserve input -> middle and all DN inputs.
            remove = (pre_pop.isin(INTERMEDIATES) & post_pop.isin(INTERMEDIATES | {"HS", "H2"})).to_numpy()
        elif control == "feedforward_chemical_only":
            retain = (pre_pop.isin({"HS", "H2"}) & post_pop.isin(INTERMEDIATES | {"DNp15"})) | (pre_pop.isin(INTERMEDIATES) & post_pop.eq("DNp15"))
            remove = (~retain).to_numpy()
        if remove.any():
            coo = chemical.tocoo()
            removed_pairs = set(zip(edges.loc[remove, "post_body"], edges.loc[remove, "pre_body"]))
            keep = np.fromiter(((ids[r], ids[c]) not in removed_pairs for r,c in zip(coo.row,coo.col)), bool, count=len(coo.data))
            chemical = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=chemical.shape)
        body_index = {int(value): i for i,value in enumerate(ids)}
        pairs = []
        for pair in electrical_pairs:
            if not pair.get("provenance") or not pair.get("evidence"):
                raise ValueError("electrical interaction lacks provenance")
            first, second = int(pair["first"]), int(pair["second"])
            if first not in body_index or second not in body_index:
                raise ValueError("electrical endpoint outside node axes")
            a,b = metadata.loc[first],metadata.loc[second]
            kinds = {a.type,b.type}
            supported = (kinds == {"HSE","H2"} and a.somaSide != b.somaSide) or (kinds == {"HSN","DNp15"} and a.somaSide == b.somaSide)
            if not supported:
                raise ValueError("electrical pair is not among the specific supported pairings")
            pairs.append([body_index[first], body_index[second]])
        pairs_array = np.asarray(pairs, dtype=np.int64).reshape(-1,2)
        if control in {"no_electrical", "feedforward_chemical_only"} or parameters.gap_conductance == 0:
            pairs_array = np.empty((0,2), dtype=np.int64)
        coupling = SparseDiffusiveCoupling.from_pairs(len(ids), pairs_array,
                     np.full(len(pairs_array), parameters.gap_conductance))
        return cls(nodes, chemical.tocsr(), coupling, parameters, control, int(remove.sum()))

    def simulate(self, drive: np.ndarray, *, initial_state: np.ndarray | None = None) -> np.ndarray:
        """Return pre-update samples at t=k*dt; input k drives k -> k+1."""
        values = np.asarray(drive, dtype=np.float64)
        n = len(self.nodes)
        if values.ndim != 2 or values.shape[1] != n or len(values) < 1 or not np.isfinite(values).all():
            raise ValueError("drive must be a finite time-by-neuron matrix")
        state = np.zeros(n) if initial_state is None else np.asarray(initial_state,dtype=np.float64).copy()
        if state.shape != (n,) or not np.isfinite(state).all():
            raise ValueError("invalid initial state")
        recorder = StateRecorder.create(self.nodes.bodyId, self.nodes.bodyId, n_steps=len(values), dtype=np.float64)
        step = self.parameters.dt_ms/self.parameters.tau_ms
        for k in range(len(values)):
            recorder.record(k, state)
            state += step * (-state + self.chemical @ np.maximum(state,0.) + self.electrical.operator @ state + values[k])
        if not np.isfinite(recorder.values).all() or not np.isfinite(state).all():
            raise FloatingPointError("nonfinite recurrent dynamics")
        return recorder.values

    def transition(self, active: np.ndarray) -> np.ndarray:
        slope = np.asarray(active, dtype=np.float64)
        if slope.shape != (len(self.nodes),) or not np.isfinite(slope).all() or np.any((slope<0)|(slope>1)):
            raise ValueError("invalid rectifier derivative")
        feedback = self.chemical.toarray()*slope[None,:] + self.electrical.operator.toarray()
        return effective_euler_transition(feedback, dt=self.parameters.dt_ms, tau=self.parameters.tau_ms)

    def preflight(self) -> dict:
        """Global contraction bound + effective Jacobians + nonlinear decay."""
        n = len(self.nodes)
        h = self.parameters.dt_ms/self.parameters.tau_ms
        base = sparse.eye(n,format="csr") + h*(self.electrical.operator-sparse.eye(n,format="csr"))
        bound = float(np.max(np.asarray(abs(base).sum(axis=1)).ravel()+h*np.asarray(abs(self.chemical).sum(axis=1)).ravel()))
        transitions = [self.transition(np.zeros(n)),self.transition(np.ones(n))]
        rho = max(spectral_radius(matrix) for matrix in transitions)
        steps = int(np.ceil(2000/self.parameters.dt_ms))+1
        drive = np.zeros((steps,n))
        perturbations = [np.ones(n),-np.ones(n),np.linspace(-1,1,n),np.random.default_rng(0).normal(size=n)]
        ratios = [float(np.linalg.norm(self.simulate(drive,initial_state=x)[-1],ord=np.inf)/np.linalg.norm(x,ord=np.inf)) for x in perturbations]
        equal_current = float(np.linalg.norm(self.electrical.current(np.ones(n))))
        negative_electrical_eigenvalue = float(np.linalg.eigvalsh(self.electrical.operator.toarray()).min())
        passed = bound < 1 and rho < 1 and max(ratios) < 1e-8 and equal_current < 1e-12
        return {"passed": bool(passed), "global_euler_lipschitz_inf_bound": bound,
                "baseline_one_sided_effective_transition_rho_max": rho,
                "zero_input_2000ms_final_initial_inf_ratios": ratios,
                "equal_state_gap_current_norm": equal_current,
                "electrical_min_eigenvalue": negative_electrical_eigenvalue,
                "electrical_pair_count": self.electrical.pair_count,
                "removed_chemical_edges": self.removed_chemical_edges,
                "stability_scope": "contraction bound covers every rectifier mask globally; reported rho is effective Euler, not raw weights"}


def controlled_drive(network: BinocularNetwork, eye_motion: list[float]) -> tuple[np.ndarray,np.ndarray]:
    motion = np.asarray(eye_motion,dtype=float)
    if motion.shape != (2,) or not np.isfinite(motion).all():
        raise ValueError("eye motion must be a finite [L,R] vector")
    p = network.parameters
    time = np.arange(round(p.duration_ms/p.dt_ms))*p.dt_ms
    nodes = network.nodes
    side = np.where(nodes.somaSide.eq("L"),0,1)
    input_gain = np.where(nodes.stage.eq("HS"),1.,np.where(nodes.stage.eq("H2"),-1.,0.))
    vector = motion[side]*input_gain*p.input_amplitude
    drive = ((time>=p.onset_ms)&(time<p.offset_ms))[:,None]*vector[None,:]
    return time,drive


def discrimination_index(translation: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    first,second = np.broadcast_arrays(np.asarray(translation,dtype=float),np.asarray(yaw,dtype=float))
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("responses must be finite")
    denominator = abs(first)+abs(second)
    return np.divide(first-second,denominator,out=np.full(first.shape,np.nan),where=denominator>0)


def response_table(network: BinocularNetwork, time: np.ndarray, traces: dict[str,np.ndarray],
                   window_ms: list[float]) -> pd.DataFrame:
    """Raw signed states and rectified emission, including both eye orders."""
    mask = (time>=window_ms[0])&(time<window_ms[1])
    if not mask.any():
        raise ValueError("response window has no samples")
    out = network.nodes[["bodyId","type","somaSide","stage"]].copy()
    for condition,trace in traces.items():
        if trace.shape != (len(time),len(out)) or not np.isfinite(trace).all():
            raise ValueError("trace/time/node axes mismatch")
        out[condition] = trace[mask].mean(axis=0)
        out[condition+"_emission"] = np.maximum(trace[mask],0.).mean(axis=0)
    # Orient each ROI-equivalent response by published upstream preference,
    # not by choosing whichever modeled output maximizes the desired result.
    # H2 prefers B on its dendritic eye; other populations use ipsilateral HS
    # preference. The record also preserves all un-oriented raw conditions.
    left = out.somaSide.eq("L").to_numpy()
    h2 = out.stage.eq("H2").to_numpy()
    prefer_left_f = left ^ h2
    yaw = np.where(prefer_left_f,out.yaw_L_F-out.yaw_R_F,out.yaw_R_F-out.yaw_L_F)
    # Figure 3e in the authors' analysis reverses the translation contrast
    # for both H2axon and H2rnaxon, but reverses yaw only for H2axon.
    # This is an analysis convention, not a preferred-direction rule in
    # the network. Preserve signed OF values alongside the index.
    reverse_translation = h2 | out.stage.eq("H2rn").to_numpy()
    translation = np.where(reverse_translation,out.translation_B-out.translation_F,out.translation_F-out.translation_B)
    out["of_yaw"] = yaw
    out["of_translation"] = translation
    out["di"] = discrimination_index(translation,yaw)
    out["magnitude_di"] = discrimination_index(abs(translation),abs(yaw))
    return out


def matched_upstream_comparison(table: pd.DataFrame) -> list[dict]:
    """Magnitude comparisons that cannot benefit from sign cancellation.

    Match each DN to its ipsilateral HS and contralateral-dendritic H2.
    Pool upstream units only after squaring each unit's signed response.
    No fitted decoder, class weights, or preferred-direction selection.
    """
    rows = []
    def rms(values):
        return float(np.sqrt(np.mean(np.square(np.asarray(values,dtype=float)))))
    for dn in table[table.stage.eq("DNp15")].itertuples(index=False):
        side = dn.somaSide
        upstream = table[(table.stage.eq("HS") & table.somaSide.eq(side)) | (table.stage.eq("H2") & table.somaSide.ne(side))]
        if len(upstream) != 4 or upstream.stage.value_counts().to_dict() != {"HS": 3, "H2": 1}:
            raise ValueError("matched upstream requires three ipsilateral HS and one contralateral H2")
        uy = rms(upstream.of_yaw); ut = rms(upstream.of_translation)
        ur = max(rms(upstream.yaw_L_F),rms(upstream.yaw_R_F))
        uc = max(rms(upstream.translation_F),rms(upstream.translation_B))
        dy = abs(float(dn.of_yaw)); dt = abs(float(dn.of_translation))
        dr = max(abs(dn.yaw_L_F),abs(dn.yaw_R_F))
        dc = max(abs(dn.translation_F),abs(dn.translation_B))
        def ratio(numerator, denominator):
            return numerator/denominator if denominator>0 else float("nan")
        rows.append({"bodyId": int(dn.bodyId), "somaSide": side,
            "matched_upstream_body_ids": upstream.bodyId.astype(int).tolist(),
            "upstream_yaw_of_rms": uy,"upstream_translation_of_rms": ut,
            "DNp15_yaw_of_magnitude": dy,"DNp15_translation_of_magnitude": dt,
            "upstream_translation_yaw_ratio": ratio(ut,uy),"DNp15_translation_yaw_ratio": ratio(dt,dy),
            "upstream_common_mode_yaw_ratio": ratio(uc,ur),"DNp15_common_mode_yaw_ratio": ratio(dc,dr),
            "declared_preferred_yaw_state": float(dn.yaw_L_F if side=="L" else dn.yaw_R_F),
            "declared_preferred_yaw_of": float(dn.of_yaw)})
    return rows


def evaluate_transformation(table: pd.DataFrame, *, epsilon: float) -> dict:
    """Predeclared magnitude + polarity contracts; signed DI cannot suffice."""
    matched = matched_upstream_comparison(table)
    if len(matched) != 2 or {r["somaSide"] for r in matched} != {"L","R"}:
        raise ValueError("primary comparison requires the bilateral DNp15 pair")
    dn_magnitude = float(table[table.stage.eq("DNp15")].magnitude_di.mean())
    upstream_magnitude = {p: float(table[table.stage.eq(p)].magnitude_di.mean()) for p in ["HS","H2"]}
    enhanced = bool(np.isfinite(dn_magnitude) and all(np.isfinite(v) and dn_magnitude+epsilon<v for v in upstream_magnitude.values()))
    per_side = []
    for row in matched:
        ratio_keys = ["upstream_translation_yaw_ratio","DNp15_translation_yaw_ratio","upstream_common_mode_yaw_ratio","DNp15_common_mode_yaw_ratio"]
        defined = all(np.isfinite(row[k]) for k in ratio_keys)
        polarity = row["declared_preferred_yaw_state"]>epsilon and row["declared_preferred_yaw_of"]>epsilon
        reduced = defined and row["DNp15_translation_yaw_ratio"]+epsilon<row["upstream_translation_yaw_ratio"] and row["DNp15_common_mode_yaw_ratio"]+epsilon<row["upstream_common_mode_yaw_ratio"]
        per_side.append({**row, "rotational_response_preserved": bool(polarity),
            "translation_and_common_mode_reduced_vs_matched_upstream": bool(reduced),
            "passed": bool(polarity and reduced)})
    return {"DNp15_mean_magnitude_di": dn_magnitude,"upstream_mean_magnitude_di": upstream_magnitude,
            "DNp15_magnitude_preference_exceeds_both_upstream_populations": enhanced,
            "per_side": per_side,"limited_neural_transformation_gate": bool(enhanced and all(r["passed"] for r in per_side)),
            "gate_is_not_full_physiological_validation": True}
