"""Sealed A-line: full-pool, scalar affine-IID, endpoint risk support.

This module never imports a downstream model, reads MOS, changes the pool,
trains a model, chooses an L hyperparameter, or uses a time/workload stop.
Guarantee: descriptive/asymptotic endpoint risk support under its assumptions;
NOT distribution-free finite-sample certification or a SRCC/PLCC guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

VERSION = '1.0.0-a-fixedpool-l3over2'
API_VERSION = 'roa-a-api-1'
PARAMETERS = {
    'reference_ratio_numerator': 3, 'reference_ratio_denominator': 2,
    'family_alpha': 0.05, 'rho_rule': 'HORIZON_ANALYTIC',
    'risk_threshold': 0.0, 'candidate_seed': 20260816,
    'candidate_domain': 'ALL_PREFIXES_1_TO_N_MINUS_1',
    'candidate_family': 'SCRAMBLED_SOBOL_PREFIX',
    'endpoint_only': True, 'early_stop': False,
    'evidence_grade': 'ASYMPTOTIC_ENDPOINT_SUPPORT_NOT_FINITE_SAMPLE_CERTIFICATE'
}

def parameters() -> dict:
    return dict(PARAMETERS)

@dataclass(frozen=True)
class Contract:
    native_n: int
    contract_id: str
    scalar_arithmetic_mean: bool
    native_iid: bool
    candidate_reference_independent: bool
    nested_prefix: bool
    online_costs: tuple[float, ...] | None = None
    cost_unit: str = 'primitive_observations'
    def __post_init__(self):
        if isinstance(self.native_n, bool) or not isinstance(self.native_n, int) or self.native_n < 2:
            raise ValueError('NATIVE_N_MUST_BE_INTEGER_AT_LEAST_2')
        if not self.contract_id or not all((self.scalar_arithmetic_mean, self.native_iid,
                  self.candidate_reference_independent, self.nested_prefix)):
            raise ValueError('UNSUPPORTED_AFFINE_IID_NESTED_MEAN_CONTRACT')
        if self.online_costs is not None:
            if len(self.online_costs) != self.native_n or any(not math.isfinite(v) or v <= 0 for v in self.online_costs):
                raise ValueError('INVALID_DECLARED_ONLINE_COSTS')
    @property
    def reference_count(self) -> int:
        return (3*self.native_n+1)//2
    @property
    def costs(self) -> tuple[float, ...]:
        return self.online_costs or tuple(float(b) for b in range(1, self.native_n+1))

@dataclass(frozen=True)
class DiscoveryPool:
    image_ids: tuple[str, ...]
    protocol_id: str
    data_role: str
    model_has_seen_pool: bool
    expected_protocol_count: int
    def __post_init__(self):
        if self.data_role not in ('train_discovery','validation_discovery','development_panel'):
            raise ValueError('TEST_IS_NOT_DISCOVERY')
        if not self.protocol_id or len(self.image_ids)<3 or len(set(self.image_ids))!=len(self.image_ids):
            raise ValueError('POOL_REQUIRES_UNIQUE_IDS_AND_AT_LEAST_3')
        if any(not isinstance(s,str) or not s for s in self.image_ids):
            raise ValueError('INVALID_IMAGE_ID')
        if len(self.image_ids)!=self.expected_protocol_count:
            raise ValueError('POOL_COUNT_DOES_NOT_MATCH_PREDECLARATION')

def horizon_parameters(m:int, candidates:int) -> dict:
    if m<3 or candidates<1:
        raise ValueError('INVALID_HORIZON')
    a=PARAMETERS['family_alpha']/(2*candidates)
    target=2*math.log(1/a)
    lo,hi=0.,max(1.,2*target)
    while hi-math.log1p(hi)<target: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if mid-math.log1p(mid)<target: lo=mid
        else: hi=mid
    x=(lo+hi)/2
    return {'planned_M':m, 'candidates':candidates,'family_alpha':PARAMETERS['family_alpha'],
            'alpha_each_side':a,'x':x,'rho':math.sqrt(x/m),'critical_factor':math.sqrt(x+1)}

def reference_risks(native:Sequence[float], candidate:Sequence[float], native_n:int) -> np.ndarray:
    y=np.asarray(native,dtype=np.float64);p=np.asarray(candidate,dtype=np.float64)
    if y.ndim!=1 or p.ndim!=1 or len(y)<2 or len(p)!=native_n-1:
        raise ValueError('INVALID_OBSERVATION_SHAPE')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('NONFINITE_OBSERVATION')
    c=np.cumsum(p,dtype=np.float64)/np.arange(1,native_n,dtype=np.float64)
    return (c-y.mean())**2-(1/len(y)+1/native_n)*y.var(ddof=1)

class FixedPoolROA:
    """Streaming moments, no output policy before complete pool. Durable raw
    storage is supplied by the execution layer and replayed through observe()."""
    def __init__(self, contract:Contract, pool:DiscoveryPool):
        self.contract,self.pool=contract,pool
        self._ids=set(pool.image_ids);self._seen=set();self._mean=np.zeros(contract.native_n-1)
        self._m2=np.zeros(contract.native_n-1);self._result=None
        self._hp=horizon_parameters(len(pool.image_ids),contract.native_n-1)
    @property
    def images_used(self): return len(self._seen)
    def plan(self,image_id:str)->dict:
        if self._result is not None: raise RuntimeError('POLICY_ALREADY_FROZEN')
        if image_id not in self._ids: raise ValueError('IMAGE_OUTSIDE_PREDECLARED_POOL')
        if image_id in self._seen: raise ValueError('DUPLICATE_DISCOVERY_IMAGE')
        return {'image_id':image_id,'reference_count':self.contract.reference_count,
                'candidate_prefix_width':self.contract.native_n-1}
    def observe(self,image_id:str,native:Sequence[float],candidate:Sequence[float])->None:
        self.plan(image_id)
        if len(native)!=self.contract.reference_count: raise ValueError('REFERENCE_COUNT_MISMATCH')
        d=reference_risks(native,candidate,self.contract.native_n)
        n=self.images_used+1;delta=d-self._mean;newmean=self._mean+delta/n
        self._m2 += delta*(d-newmean);self._mean=newmean;self._seen.add(image_id)
    def progress(self)->dict:
        return {'images_used':self.images_used,'planned_images':len(self.pool.image_ids),
                'final_B':None,'decision_ready':self.images_used==len(self.pool.image_ids),
                'partial_evidence_is_not_final':True}
    def finalize(self)->dict:
        if self._result is not None:
            import copy
            return copy.deepcopy(self._result)
        if self.images_used!=len(self.pool.image_ids):
            raise RuntimeError('FULL_POOL_REQUIRED_BEFORE_POLICY_FREEZE')
        n=self.images_used;N=self.contract.native_n
        sd=np.sqrt(np.maximum(0,self._m2/(n-1)));w=self._hp['critical_factor']*sd/math.sqrt(n)
        lo,up=self._mean-w,self._mean+w
        valid=sd>0 # empirical constancy is not a proof of zero population variance
        supported=[int(i+1) for i in np.flatnonzero(valid & (up<0))]
        excluded=[int(i+1) for i in np.flatnonzero(valid & (lo>0))]
        costs=self.contract.costs
        selected=min(supported+[N],key=lambda b:(costs[b-1],b!=N,b))
        def vals(z):return [float(v) if ok else None for v,ok in zip(z,valid)]
        self._result={
            'core_version':VERSION,'api_version':API_VERSION,'parameter_lock':parameters(),
            'selected_B':selected,'native_N':N,'reference_L':self.contract.reference_count,
            'M_used':n,'M_planned':len(self.pool.image_ids),'M_learned':False,
            'supported_B':supported,'excluded_B':excluded,'risk_mean':self._mean.tolist(),
            'risk_sd':sd.tolist(),'lower':vals(lo),'upper':vals(up),'evidence':dict(self._hp),
            'online_cost':costs[selected-1],'native_online_cost':costs[-1],
            'online_cost_unit':self.contract.cost_unit,'observation_reduction':1-selected/N,
            'cost_reduction':1-costs[selected-1]/costs[-1],
            'reference_primitives':n*self.contract.reference_count,'candidate_primitives':n*(N-1),
            'discovery_primitives':n*(self.contract.reference_count+N-1),
            'stop_reason':'COMPLETE_PREDECLARED_POOL','early_stop_enabled':False,
            'status':'REDUCED_SUPPORTED' if selected!=N else 'NATIVE_FALLBACK',
            'optimality_scope':'MINIMUM_COST_WITHIN_ENDPOINT_SUPPORTED_SET',
            'generalization_certificate':False,'predictive_SRCC_PLCC_guaranteed':False,
            'evidence_grade':PARAMETERS['evidence_grade'],
            'discovery_role':self.pool.data_role,'model_has_seen_discovery':self.pool.model_has_seen_pool}
        return self.finalize()
