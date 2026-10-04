"""Build traceable measured tables. Scientific interpretation is in summary.md."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;O=H/'reports'


def read(name):
    p=O/name
    return pd.read_csv(p) if p.exists() and p.stat().st_size>1 else pd.DataFrame()


def fmt(v):
    if isinstance(v,str):return v
    if isinstance(v,(int,np.integer)):return str(v)
    if np.isfinite(v) and 0<abs(v)<.001:return f'{v:.3e}'
    return f'{v:.3f}' if np.isfinite(v) else 'N/A'


def table(df,cols):
    if df.empty:return '未実行または集計対象なし.\n'
    return '\n'.join(['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']+
                     ['|'+'|'.join(fmt(row[c]) for c in cols)+'|' for _,row in df.iterrows()])+'\n'


def interval(r):return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def primary_table():
    d=read('primary_means.csv');rows=[]
    methods=['TINS','static','M0_pt','M1_pt','L0_raw','L1_raw','L2_raw','L0_norm','L1_norm','L2_norm',
             'L0_cdf','L2_cdf','L0_plp','L1_plp','L2_plp','class_L2_raw','class_L2_norm','class_L2_plp',
             'REPRISE_L0_M0','REPRISE_L1_M0','REPRISE_L2_M0','REPRISE_L1_M1','REPRISE_L2_M1']
    if d.empty:return ''
    for method in methods:
        for suffix in ['', '+TINS']:
            if method=='TINS' and suffix:continue
            g=d[(d.method==method+suffix)&(d.n==5)]
            if g.empty:continue
            row={'method':method+suffix}
            for stream in ['near','far']:
                for metric in ['AUROC','FPR95']:
                    r=g[(g.stream==stream)&(g.metric==metric)]
                    row[stream+' '+metric]=interval(r.iloc[0]) if len(r) else '未完了'
            rows.append(row)
    return table(pd.DataFrame(rows),['method','near AUROC','near FPR95','far AUROC','far FPR95'])


def phase1():
    d=read('paired_differences.csv');c=read('candidate_vs_no_memory_lp.csv');cap=read('capacity_paired_differences.csv')
    title='# Phase 1: 同一グラフでの性能比較\n\n'
    text='評価は既使用dev1. ImageNet train内のID900クラス/18,000枚, nearはtrain内の残り100クラス/5,000枚, farはOpenImage-O val/1,763枚. OpenOOD test平均とは別の母集団である. 支持12+較正4/クラス, B/14+L/14, batch256. 全方式が同じ抽出・順序・グラフを共有する.\n\n'
    text+='以下の区間は3順序を各抽出内で平均してから5抽出で作る95% t区間. 固定した画像・クラスの下での抽出/順序の近似不確実性であり, 未知クラス全般や選択後の確認的被覆を保証しない. 15条件を独立n=15とは扱わない. 単位は%またはpercentage points. FPR95は低いほど良い.\n\n'
    text+='## 同一設定の主表\n\n'+primary_table()
    text+='\nL0=過去uから15回(初回はy), L1=全節点0から15回, L2=相対残差1e-8まで. raw=u, norm=u/支持u中央値, cdf=到着前に固定した同じ較正集合のCDF, plp=現在の較正順位. class_L2は900クラスの収束伝播の最大値. M0は較正再利用, M1は1枚/クラスずつ4役割へ分割. スコアはビュー間の積. TINSとの積をp値とは呼ばない.\n\n'
    text+='## 対応のある差\n\n差はa−b. FPR95の負値はaが良い. 選択のない機構比較とdev選択後の比較を分ける.\n\n'
    cols=['a','b','stream','metric','mean','lo','hi','n']
    if not d.empty:text+=table(d[d.n==5],cols)
    text+='\n## メモリを持たないLP全体との探索的比較\n\nここではcurrent-calibration pLPも候補に含める. 事前登録した単純変換raw/norm/frozen-CDFの比較表は `candidate_vs_selected_lp.csv` に別保存. 選択にも評価にもdev1を使っているため, 未使用確認試験の優位性主張ではない.\n\n'
    if not c.empty:
        c=c[(c.capacity=='all')&c.a.str.endswith('@cfg0')]
        text+=table(c,cols)
    text+='\n## 容量制限\n\n支持・較正を固定し, グラフのストリーム節点とA1,A2,Mを各8,192枚以下にした. 差は容量8192−全履歴. 同バイト数とは限らない.\n\n'
    if not cap.empty:
        take=['L2_raw','L2_plp','class_L2_raw','REPRISE_L0_M0','REPRISE_L1_M1','REPRISE_L2_M1']
        text+=table(cap[cap.method.isin(take+[x+'+TINS' for x in take])],['method','stream','metric','mean','lo','hi','n'])
    text+='\n## その他の比較と診断\n\n`matched_resource_comparison.csv` は機構表から分けた全構成の生指標. `selected_tuned_methods.csv` の completed_configs が3未満の方式は調整未完了. `baseline_scope.md` に記す通り, AdaNeg_type/OODD_typeは簡略メモリ比較であり, 原法の再現とは呼ばない.\n\n'
    text+='`rank_scale_diagnostics.csv` はバッチ内の順位反転・同点, 較正分位点, 支持尺度. `single_view_rank_metrics.csv` は1ビュー内の全体AUROCと同一バッチID/OOD対で重み付けしたAUROC. 単一ビュー/同一バッチの単調な順位変換は同点以外の順位情報を増やさない. 2ビュー積やTINSとの積では非線形変換により結合後の順位も変わるため, 差を共形保証の効果と即断しない. `numeric_score_audit.csv` と全画像のlogスコアを保存. `frozen_v5_equivalence.csv` は旧CPU実装の保存値との数値差. 現行GPUグラフと旧結果のbitwise一致は主張しない.\n'
    text+='\n## 事後診断: 静的因子と適応メモリを分ける\n\nREPRISE−pLPは静的距離の因子も含む. 主結果閲覧後にこの交絡を明記し, 保存済みスコアから static-p × pLP を再集計した. 新たな候補選択には使わず, 確認的比較とも呼ばない. 登録時点は `provenance/static_factor_diagnostic_registration.json`.\n\n'
    sf=read('static_factor_paired.csv')
    if not sf.empty:text+=table(sf[sf.a.str.startswith('REPRISE_')&sf.a.str.endswith('+TINS')],cols)
    text+='\nこの対照ではREPRISEの適応メモリはnearを改善する一方, farでは悪化した. pLP単独との比較がfarでも改善していても, メモリ適応そのものがfarに有利とは言えない.\n'
    (H/'phase1.md').write_text(title+text)


def phase2():
    cal=read('calibration_audit.csv');p=read('calibration_paired_differences.csv');ss=read('solver_summary.csv');sp=read('solver_paired_time.csv')
    text='# Phase 2: 較正と反復方式の監査\n\n経験的頻度と数学的保証を別々に記す. 区間は画像を独立二項試行とした区間ではなく, 合成の独立再生成200回または実特徴の較正再抽出50回を単位とする95% t区間. 実特徴は支持と評価画像を固定した同一dev1内の診断.\n\n'
    text+='## 名目alpha=.10のID下側確率\n\n下表のmean/lo/hiは%表示. alpha=.01/.05, 早期/後期, クラス別は `calibration_audit.csv` に全件保存. クラス別の観測値をクラス条件付き保証とは呼ばない.\n\n'
    if not cal.empty:
        g=cal[(cal.alpha==.1)&(cal.window=='all')&cal.component.isin(['static','M0_pt','M1_pt','L0_plp','L1_plp','L2_plp'])].copy()
        for key in ['mean','lo','hi']:g[key]*=100
        text+=table(g,['stage','view','condition','component','mean','lo','hi','n'])
    text+='\n## 対応のある較正確率差\n\n差はa−b, percentage points. 小さいことだけを検出能力の改善とは解釈しない. M1の合成較正は役割ごと25枚なので最小p=1/26. alpha=.01で0件になるのは分解能による. 実特徴では役割ごと900枚, 最小p=1/901.\n\n'
    if not p.empty:text+=table(p[(p.alpha==.1)&(p.window=='all')],['stage','view','condition','a','b','mean','lo','hi','n'])
    text+='\n## LP計算\n\n時間は2ビュー合計. 特徴抽出・TINS推論を除いたキャッシュ後の処理. solve_mean_batch_msはbatch256の解法部分, cache_mean_batch_msは共有グラフ構築を加えた値. メモリ処理や固定グラフ初期構築を含むend-to-end値ではない. GPU4台並行時の実測であり, バッチ充足待ちやbatch1応答遅延ではない.\n\n'
    if not ss.empty:text+=table(ss[ss.metric.isin(['solve_seconds','solve_mean_batch_ms','cache_mean_batch_ms','iterations_mean','relative_residual_max'])],['stream','solver','metric','mean','lo','hi','n'])
    text+='\n対応のある解法時間差(a−L0).\n\n'
    if not sp.empty:text+=table(sp,['stream','a','b','metric','mean','lo','hi','n'])
    text+='\nL0/L1の未収束表示は, 意図的な有限15反復を収束解と誤認しないための診断. L2/class_L2の未収束は別途失敗扱いで点検する. `solver_score_changes.csv` にu/p差と順位相関, `solver_audit.csv` に全バッチの残差・時間を保存.\n\n'
    text+='## 証明の成立範囲\n\n`validity_lemmas.md` に仮定付きの証明を記載した. 交換可能な較正/queryと置換同変な計算なら有限15反復でも順位保証は可能. L1を反復数だけで否定しない. 一方, L0の履歴依存, kNN同点処理, クラス固定枚数抽出, 設定選択履歴は別の問題である. 実際のREPRISE全体の共形保証は未証明.\n\n'
    text+='較正/新着の役割交換後に履歴を再実行すると, 合成診断のu差はL0で約2.54e-4, L1/L2で約1e-7以下. 同一特徴のkNN同点では節点置換によるグラフ差約.158も確認. これらは対称性の診断であり, 誤警報率への直接の誤差保証ではない. ログのboundary_tiesは初期/新着行のtop-k境界を数え, 過去行のmergeに生じた全ての同点を網羅する計数ではない.\n\n'
    text+='独立IIDの新しい4役割較正, 独立に凍結した設定, 外生IIDストリーム等の追加仮定なら, M1の入口と読出しに周辺保証を導ける. 実際の固定枚数/既使用プールの設定はそのまま当てはまらない. 900クラス×4枚の層化較正をIID扱いすると一般的順位保証が破れる解析的反例も記した. p値の積とTINSとの積にはp値保証が引き継がれない. FDRや全時刻同時保証もない.\n\n'
    text+='合成のclass_burstはID/OODの到着位置も変わる複合ストレス. 実特徴class_burstはOOD位置を固定してIDの並びだけを変える. 実特徴の分布ずれ条件は今回未実施であり, 合成のID_shiftと混同しない.\n'
    (H/'phase2.md').write_text(text)


def plots():
    d=read('primary_means.csv')
    if d.empty or not (d.n==5).any():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['L2_raw','L2_norm','L2_cdf','L2_plp','class_L2_raw','class_L2_plp','M0_pt','REPRISE_L0_M0','REPRISE_L1_M1']
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for i,suffix in enumerate(['','+TINS']):
        for j,stream in enumerate(['near','far']):
            ax=axes[i,j]
            for k,name in enumerate(names):
                g=d[(d.method==name+suffix)&(d.stream==stream)&(d.metric=='FPR95')&(d.n==5)]
                if len(g):
                    r=g.iloc[0];ax.errorbar(r['mean'],k,xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',color='#216e77',capsize=3)
            ax.set_title(stream+' / '+(suffix or 'standalone'));ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.2);ax.set_xlabel('FPR95 (%) - lower is better')
            ax.set_ylim(len(names)-.5,-.5)
    fig.suptitle('Used development data | matched 5 draws x 3 orders | draw-level 95% t intervals')
    fig.tight_layout();fig.savefig(O/'primary_comparison.pdf');fig.savefig(O/'primary_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':phase1();phase2();plots()
