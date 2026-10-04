import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent
out=H/'preregistration.json'
if out.exists():raise FileExistsError(out)
pr={
 'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Phase 0-2; all new selection on previously used dev1 only',
 'budget_gpu_hours':8,'resource_accounting':'Conservative process wall seconds times visible allocated GPUs, including pilots and failed runs. CPU-only jobs report CPU wall separately.',
 'paper_sha256':'784f7ee7f574b977625d930a4aaaa572ef7bfcfc8b2d74f102bab7f2dd82aa27',
 'base_repository_commit':'965e74420e6395cda902486aa62ed8fe04d6bcae',
 'settings':{'n0':48,'K':20,'m':1,'gamma':1.,'alpha':.9,'kg':10,'support_per_class':12,'cal_per_class':4,'batch':256,'iterations':15},
 'thresholds_exact':[.3,.2,.10191613435745239], 'epsilon3_paper_display':.102,
 'epsilon3_note':'Primary reproduction uses serialized exact final value; report .102 nominal separately, never silently substitute.',
 'primary':{'views':['B14','L14'],'draws':list(range(5)),'seeds':[123,124,125],'streams':['near','far'],
            'support_draws':'existing distinct audited R5 draws, common to every method','stream_images':'all existing dev1 stream images; exact cached TINS order',
            'solvers':['L0_legacy_warm15','L1_all_zero15','L2_relative_equation_residual_1e-8'],
            'scores':['static_p','memory_pt_M0','memory_pt_M1','raw_u','support_median_normalized_u','frozen_CDF','current_calibration_pLP','REPRISE'],
            'classwise':'all 900 ID classes, chunked FP64 convergence; max_c U_c; same four score transformations',
            'fusion':['standalone','times_same_draw_same_stream_TINS'],
            'frozen_CDF':'Per draw/view/solver, CDF of the same 4 calibration images/class on the support+calibration graph before any stream arrival; frozen thereafter; no extra labels.',
            'normalization_floor':1e-12,'log_score_floor':1e-300},
 'metrics':{'primary':'near FPR95','secondary':['near AUROC','far FPR95','far AUROC'],
            'FPR95':'At least 95% ID accepted; accept score >= lower empirical quantile; all threshold ties accepted; no interpolation; actual acceptance and tie counts recorded.',
            'AUROC':'ID-positive average ranks for ties; sklearn roc_auc_score',
            'macro':'Existing dev near contains 100 unknown ImageNet classes in one mixed stream, far OpenImage-O val; not OpenOOD test macro.',
            'CI':'Paired differences, average three orders within each support/calibration draw, 95% t interval df=4. Fixed image/class population. Exploratory dev intervals, no confirmatory claim.',
            'confirmation_reserved':'10 genuinely distinct support/calibration draws x3 orders; not run/opened now; two tests candidate-vs-strongest LP and candidate-vs-no-entrance, Holm.'},
 'candidate_cap':4,'candidates':['M0+L1','M0+L2','M1+L1','M1+L2'],
 'selection':'dev1 minimum near FPR95 subject to far worsening <= .5 percentage points; -2 point near improvement is an exploratory target, not acceptance criterion or promise.',
 'equal_additional_tuning_budget':{'configurations_per_family':3,'maximum_authorized':24,
      'LP_and_REPRISE':[{'kg':10,'alpha':.9},{'kg':10,'alpha':.8},{'kg':20,'alpha':.9}],
      'Mahalanobis++_type_shrinkage':[.01,.5,.9],
      'AdaNeg_type':[{'tau':.45,'weight':1.},{'tau':.55,'weight':2.},{'tau':.65,'weight':1.}],
      'OODD_type':[{'dictionary':1024,'weight':1.},{'dictionary':4096,'weight':1.},{'dictionary':8192,'weight':1.}],
      'note':'Prior REPRISE and baseline search disclosed separately. No outcome-driven extension. All 3 configurations evaluated on the same 15 draw/order pairs if budget permits; incomplete families cannot be called tuned strongest.'},
 'capacities':['all_history',8192],
 'simulation':{'independent_replicates':200,'seed_base':972700,'ID_classes':25,'OOD_classes':2,'dimension':16,'support_per_class':12,'cal_per_class':4,'stream_length':128,'batch':16,
     'generator':'unit-normalized Gaussian class means (seeded); feature = mean + N(0,.3^2 I); equal ID mixture; OOD means independent; no image duplication.',
     'conditions':['iid_ID_only','iid_mixed_25pct_OOD','class_balanced_cal_mixed','ID_shift_mixed','class_burst_mixed'],
     'iid_cal':'100 independent ID draws from the equal mixture; role masks every fourth iid draw; separate balanced condition uses 4/class.',
     'alpha':[.01,.05,.1],'unit':'independently regenerated support/calibration/stream, not individual images'},
 'real_calibration':{'draws':50,'support':'fixed R5 draw0, 12/class','calibration_pool':'68 remaining R5 shot images/class; 4 sampled without replacement/class each replicate',
     'ID_classes':900,'stream':'preexisting audited dev1 train audit subset: 4500 ID +1500 near OOD; ID-only and mixed and class-burst conditions',
     'views':['B14','L14'],'selection_seed':972750,'unit':'calibration resample; stream cohort fixed; no binomial independence claim'},
 'priority':['audit_and_unit_tests','primary_30_paired_runs','synthetic_validity_200','real_calibration_50','matched_tuning_and_capacity'],
 'stop':'All preregistered required Phase0-2 work complete, or explicit budget/data obstacle; never stop/search based on favorable outcomes.',
 'source_sha256':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))},
}
out.write_text(json.dumps(pr,indent=2)+'\n')
(H/'preregistration.sha256').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'\n')
(H/'config_original.yaml').write_text('# Exact serialized paper v5 configuration; JSON is also valid YAML.\n'+json.dumps({**pr['settings'],'thresholds':pr['thresholds_exact'],'paper_epsilon3_rounded':.102},indent=2)+'\n')
