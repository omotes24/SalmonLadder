"""Render the final scientific report from completed, paired result tables."""
from pathlib import Path
import json
import pandas as pd
from report import paired
H=Path(__file__).resolve().parent;O=H/'reports'


def read(name):
    p=O/name
    return pd.read_csv(p) if p.exists() and p.stat().st_size>1 else pd.DataFrame()
def val(row):return f"{row['mean']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]"
def table(rows,columns):
    return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']+['|'+'|'.join(str(r[c]) for c in columns)+'|' for r in rows])+'\n'


def main():
    execution=json.loads((O/'execution_summary.json').read_text())
    integrity=json.loads((O/'result_integrity.json').read_text())
    if integrity['errors']:raise RuntimeError(integrity['errors'])
    primary=read('primary_means.csv');runs=read('lp_baselines.csv');deltas=read('paired_differences.csv')
    def mean(method,stream,metric):
        return primary[(primary.method==method)&(primary.stream==stream)&(primary.metric==metric)].iloc[0]
    def difference(a,b,stream,metric='FPR95'):
        return val(paired(runs,a,b,stream,metric))
    text='# REPRISE Phase 0–2: 実測結果と妥当性監査\n\n'
    text+='結論は **主要貢献の一部だけを支持**. 同一設定では現在の較正順位とメモリの組合せがraw-LPおよび単純正規化を上回った. 一方, 現行構成の共形保証, 再来による因果的改善, 未知データへの一般化, 最新既存法全体への優位性は確定していない.\n\n'
    text+='## 実行範囲と統計単位\n\n'
    rows=[{'実験':k,'完了/登録':f"{v['completed']}/{v['planned']}"} for k,v in execution['completion'].items()]
    text+=table(rows,['実験','完了/登録'])
    text+=f"\n累積GPU予算の計上は **{execution['charged_gpu_hours_conservative']:.3f}/8 GPU時間**. 成功ジョブの監督側壁時計実測は {execution['measured_successful_dispatch_gpu_hours']:.3f} GPU時間. 前者はpilot等200秒と失敗ジョブの上限計上を含む. GPU演算器の稼働時間やend-to-end計算時間とは異なる.\n\n"
    text+=f"失敗した実験試行は{len(execution['failed_attempts'])}件. 画像IDのobject dtypeを読めず保存時に失敗した8件は, 同じseed・設定で修正後に再実行した. 元ログと失敗台帳を保持. 単体テスト16件合格. 失敗・未実行は `reports/execution_summary.json`, `reports/unexecuted_runs.csv`, `reports/dispatch_history/` から個別に追跡できる.\n\n"
    text+='評価は**既使用dev1**: ID900クラス18,000枚, nearはImageNet train内の100未知クラス5,000枚, farはOpenImage-O val1,763枚. 支持12+較正4枚/クラス, B/14+L/14, batch256. 5抽出×3順序を対応させた. 区間は3順序を抽出内で平均した後のn=5対応付き95% t区間. 固定した画像・クラス集合の下での抽出/順序の不確実性であり, 新規クラスへの一般化区間ではない. dev選択後の差は探索的. 新規holdoutは開封していない.\n\n'
    text+='L0=現行warm15, L1=全節点0から15回, L2=相対方程式残差1e-8まで. M0=現行較正再利用, M1=各クラスの4枚を4役割に分割. 全スコアを同一グラフで計算. 以下の値は%, 差はポイント. FPR95は低いほどよい.\n\n'
    text+='## 1. 標準LPとREPRISE\n\n'
    rows=[]
    methods=[('標準LP, 収束raw','L2_raw'),('単純正規化, L1','L1_norm'),('現在の較正順位, L1','L1_plp'),('メモリptのみ','M0_pt'),('現行REPRISE (Ours)','REPRISE_L0_M0'),('候補 L1+M0 (Ours)','REPRISE_L1_M0'),('候補 L1+M1 (Ours)','REPRISE_L1_M1')]
    for label,method in methods:
        row={'方式 (+TINS)':label}
        for stream in ['near','far']:
            for metric in ['AUROC','FPR95']:row[stream+' '+metric]=val(mean(method+'+TINS',stream,metric))
        rows.append(row)
    text+=table(rows,['方式 (+TINS)','near AUROC','near FPR95','far AUROC','far FPR95'])
    for stream in ['near','far']:
        text+=f"\n現行REPRISE−収束raw-LP, {stream}: ΔFPR95={difference('REPRISE_L0_M0+TINS','L2_raw+TINS',stream)}, ΔAUROC={difference('REPRISE_L0_M0+TINS','L2_raw+TINS',stream,'AUROC')}.\n"
    text+='\nraw/norm/frozen-CDFの同条件LPでnearが最良だったのはL1_norm. current-calibration pLPまで含めたメモリなしLP全体ではL1_plpが最良. 後者を強い探索的対照としても記す. 追加グラフ設定の調整が未完なら, 全調整済みの最強LPとは呼ばない. 900クラス別LPは全チャンネルを実行済み. 詳細は `phase1.md`, `reports/candidate_vs_selected_lp.csv`, `reports/candidate_vs_no_memory_lp.csv`.\n\n'
    text+='## 2. 共形化と簡単な正規化\n\n'
    for stream in ['near','far']:
        text+=f"収束pLP−同じ収束uの支持中央値正規化, {stream}: ΔFPR95={difference('L2_plp+TINS','L2_norm+TINS',stream)}, ΔAUROC={difference('L2_plp+TINS','L2_norm+TINS',stream,'AUROC')}.\n\n"
    text+='単一ビュー・同一バッチの順位反転は0. pLPは同点を作るが, その単位では新しい順序情報を作らない. 差はバッチ間の尺度調整および非線形なビュー結合で現れる. これを共形保証の効果と呼ぶ根拠はない. 到着前の固定CDF単独はnear/farともFPR95=100%となり, 現在のグラフに対応した尺度が必要な条件を示した. 対応付きの全体/バッチ内AUROCは `reports/single_view_paired_differences.csv`.\n\n'
    text+='スコア単独も分離した. 現行REPRISE単独のnear/far FPR95は '+val(mean('REPRISE_L0_M0','near','FPR95'))+' / '+val(mean('REPRISE_L0_M0','far','FPR95'))+'. 同じTINS乗算の効果はnear '+difference('REPRISE_L0_M0+TINS','REPRISE_L0_M0','near')+', far '+difference('REPRISE_L0_M0+TINS','REPRISE_L0_M0','far')+'. TINSが全条件で改善するとは言えない.\n\n'
    text+='## 3. 初期化・反復方式\n\n'
    for mode in ['L1','L2']:
        text+=f"{mode}のpLP−L0のpLP (+TINS): near ΔFPR95={difference(mode+'_plp+TINS','L0_plp+TINS','near')}, far ΔFPR95={difference(mode+'_plp+TINS','L0_plp+TINS','far')}.\n\n"
    time=read('solver_summary.csv');tp=read('solver_paired_time.csv');rows=[]
    for mode in ['L0','L1','L2']:
        g=time[(time.stream=='near')&(time.solver==mode)]
        rows.append({'解法':mode,'全nearストリームのsolve秒':val(g[g.metric=='solve_seconds'].iloc[0]),'batch平均solve ms':val(g[g.metric=='solve_mean_batch_ms'].iloc[0])})
    text+=table(rows,['解法','全nearストリームのsolve秒','batch平均solve ms'])
    for mode in ['L1','L2']:
        g=tp[(tp.stream=='near')&(tp.a==mode)&(tp.metric=='solve_seconds')].iloc[0]
        text+=f"\n{mode}−L0の対応付きsolve時間差: {val(g)} 秒.\n"
    text+='\n時間はキャッシュ後の2ビュー合計, 共有グラフ構築・メモリ・特徴抽出・TINS推論を含まないsolve部分. L2は全検証対象で所定残差に収束したかを `result_integrity.json` で点検した. L1は有限15反復の計算として評価し, 収束解と偽らない.\n\n'
    text+='較正率の変化は `phase2.md` に200独立合成試行と50実特徴再抽出の対応区間を掲載. 単純に低い率を良い検出と扱わず, 名目水準・OOD採用率・p値分解能を併記した.\n\n'
    calibration=read('calibration_audit.csv');cp=read('calibration_paired_differences.csv');rows=[]
    for view in ['B14','L14']:
        for condition in ['ID_only','mixed','class_burst']:
            row={'ビュー/条件':view+'/'+condition}
            for mode in ['L0','L1','L2']:
                g=calibration[(calibration.stage=='real_calibration')&(calibration.view==view)&(calibration.condition==condition)&(calibration.window=='all')&(calibration.alpha==.1)&(calibration.component==mode+'_plp')].iloc[0].copy()
                for k in ['mean','lo','hi']:g[k]*=100
                row[mode+' ID下側確率 (%)']=val(g)
            rows.append(row)
    text+='実特徴, 名目alpha=.10のPr(pLP<=alpha | ID). 支持と評価画像は固定し, 較正だけ50回再抽出.\n\n'+table(rows,['ビュー/条件','L0 ID下側確率 (%)','L1 ID下側確率 (%)','L2 ID下側確率 (%)'])
    g=cp[(cp.stage=='real_calibration')&(cp.view=='L14')&(cp.condition=='class_burst')&(cp.window=='all')&(cp.alpha==.1)&(cp.a=='L1_plp')&(cp.b=='L0_plp')].iloc[0]
    text+='\nL/14のclass_burstではL1−L0の対応差が '+val(g)+' ポイント. **全節点同時初期化でも, この実データ到着条件の較正は回復しない**. この率は既使用プール・固定評価群に条件づけた経験値であり, そのまま母集団の定理違反の証明とはしない.\n\n'
    text+='## 4. 証明できる範囲と未証明部分\n\n'
    text+='- **追加仮定付きで証明**: 較正とqueryが交換可能で, 全計算が置換同変なら順位p値は周辺でsuper-uniform. L1は15回でもこの対称性を保ち得る. 収束は必要条件ではない.\n- **追加仮定付きで証明**: 新規IIDの4役割較正, 独立に凍結した設定, 外生IIDストリーム等ならM1の入口・読出しに周辺保証を示せる. 現行の固定クラス枚数/既使用プールへそのまま適用できない.\n- **反例/非対称性あり**: 4枚/クラスの層化抽出だけではプール順位の一般的保証は出ない. L0の到着年齢依存, kNN同点の配列依存を実測. p値の未補正積は一般に有効なp値ではない.\n- **未証明**: 実際のREPRISE全体, 分布ずれ, クラス連続到着, クラス条件付き制御, FDR, 全時刻同時制御. 良好な実測頻度で証明を置き換えない.\n\n'
    text+='式・反例・一次文献・実装との対応は `validity_lemmas.md`. 残差から座標誤差と順位変化幅の条件付き上限を導いたが, 全大規模丸め行列のノルムを機械証明していないため, 数値ログを厳密な計算機証明とは呼ばない.\n\n'
    text+='## 5. 次段階候補とメモリの条件依存性\n\n'
    text+='同一設定の4登録候補ではL1+M0がnear FPR95最小. +TINSでのメモリなし最良L1_pLPに対する差はnear '+difference('REPRISE_L1_M0+TINS','L1_plp+TINS','near')+', far '+difference('REPRISE_L1_M0+TINS','L1_plp+TINS','far')+'. 開発データでの候補であり, 新規testでの確認ではない. L1+M1は較正再利用を分ける診断候補で, 性能優位や実条件の保証は主張しない.\n\n'
    text+='pt因子を掛ける効果 (REPRISE−L0 pLP, +TINS) はnear '+difference('REPRISE_L0_M0+TINS','L0_plp+TINS','near')+', far '+difference('REPRISE_L0_M0+TINS','L0_plp+TINS','far')+'. ただし, この差には静的距離を掛ける効果も含む.\n\n'
    sf=read('static_factor_paired.csv');rows=[]
    for mode in ['L0','L1']:
        row={'REPRISE−静的p×pLP (+TINS)':mode}
        for stream in ['near','far']:
            g=sf[(sf.a==f'REPRISE_{mode}_M0+TINS')&(sf.stream==stream)&(sf.metric=='FPR95')]
            row[stream+' ΔFPR95']=val(g.iloc[0])
        rows.append(row)
    text+='そこで主結果閲覧後の診断として, 同じ保存値から静的p×pLPを集計した. **適応メモリはnearで効き, farでは悪化する**. この対照は新たな候補選択に用いず, 事後診断と明記する.\n\n'+table(rows,['REPRISE−静的p×pLP (+TINS)','near ΔFPR95','far ΔFPR95'])
    text+='\nこの比較でも同じ未知クラスの再来が改善原因とは言えない. 時刻・履歴量・queryを固定するPhase3は未実行.\n\n'
    text+='実特徴のmixed/class_burstでのメモリ寄与と, ID-onlyでの較正率は `reports/real_detection_paired.csv`, `phase2.md`. 合成ID分布ずれではM0_ptの名目.10でのID下側確率が45.948% [44.778,47.117]へ増加し, 静的pの18.146% [17.041,19.250]を上回った. 役割分離だけでも解消しない. これは合成ストレスでの失敗境界であり, 実画像の分布ずれへ数値を外挿しない.\n\n'
    cap=read('capacity_paired_differences.csv')
    if not cap.empty:
        rows=[]
        for method in ['L1_plp+TINS','REPRISE_L0_M0+TINS','REPRISE_L1_M0+TINS','REPRISE_L1_M1+TINS']:
            row={'方式':method}
            for stream in ['near','far']:
                g=cap[(cap.method==method)&(cap.stream==stream)&(cap.metric=='FPR95')]
                row[stream+' ΔFPR95']=val(g.iloc[0]) if len(g) else '未完了'
            rows.append(row)
        text+='容量8192−全履歴の対応差. 支持・較正は固定し, グラフのstream節点とA1,A2,Mをそれぞれ制限した.\n\n'+table(rows,['方式','near ΔFPR95','far ΔFPR95'])
    text+='\n## 未実行と次に必要な予算\n\n'
    for stage in ['tuning','capacity','baselines']:
        v=execution['completion'][stage]
        if v['completed']<v['planned']:
            missing=[x for x in execution['unexecuted_individual_tasks'] if x['stage']==stage]
            estimate=missing[0].get('stage_estimated_gpu_seconds') if missing else None
            text+=f"- {stage}: {v['completed']}/{v['planned']}完了. "+(f"全段階の事前費用見積り{estimate/3600:.3f} GPU時間が残予算に入らず未起動. " if estimate else '個別台帳参照. ')+"未実行を成功扱いしない.\n"
    text+='- AdaNeg/OODD公式原法の忠実な再現は未完. 実行したtype実装は簡略メモリ対照. 同等の追加調整枠を登録したが, LP調整が未完なら, 基準点の同条件比較と調整済み比較を混ぜない.\n- Phase3再来の因果対照, Phase4低OOD率/ID先行/小バッチ/長期運用と生画像からの計算量, Phase5新規holdoutは今回の範囲外. 新規holdout候補は未使用と認定できていない.\n- 実特徴でのID分布ずれ, 同じID誤採用率で揃えた入口比較は未実行. 現在の名目閾値での比較から入口の普遍的必要性を断言しない.\n\n'
    text+='追加ラベルは主比較0枚. M1も既存4較正枚を分割. 実特徴50再抽出は既存の68枚/クラスの候補プールを再利用する別診断であり, 1 runの較正は4枚/クラス. 主表へ追加画像を混ぜない. 初期8 GPU時間を超える処理やholdout開封は行っていない. 次の追加LP調整と, 未使用データの使用履歴確認が残る.\n\n'
    text+='## 論文の主張への帰結\n\n'
    text+='支持されるのは, この既使用開発条件での「現在グラフに対応した尺度調整とメモリの組合せ」の有効性. 「較正分割だけで共形保証成立」「p値の積が有効なp値」「全条件で優越」「再来こそ改善原因」「使用済みtestを独立holdoutとする」主張は採用しない. 収束まで解くことが一律に性能・較正を良くするという主張も支持されない.\n\n'
    text+='コードと再現手順は `README.md`. 主設定・分割・コードのハッシュは `preregistration.json`, `data_manifest.json`, `split_manifest.json`, `provenance/`. 生スコアはhadesの `/home/omote/reprise_lp_audit_20260927/results/` に保存し, 全ファイルのSHA256を `reports/result_manifest.json` に残した. 既存の論文・実装・結果は上書きしていない.\n'
    (H/'summary.md').write_text(text)


if __name__=='__main__':main()
