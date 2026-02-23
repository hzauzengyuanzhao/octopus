#!/bin/bash
set -euo pipefail

########################################
# Usage:
# bash TAD_eval_fraction.sh \
#   GT.txt \
#   Species \
#   CHR | ALL \
#   model1.txt model2.txt ...
########################################

GT_TXT="$1"
SPECIES="$2"
CHR="$3"
shift 3
MODELS="$@"

########################################
# Parameters
########################################
FRAC_GT=0.7
FRAC_MODEL=0.7

########################################
# Output
########################################
OUT="${SPECIES}_TAD_evaluation.mutual_fraction_0.7.tsv"
echo -e "Model\tRecall\tPrecision" > "$OUT"

########################################
# Prepare GT BED (keep chr names)
########################################
awk '
BEGIN{OFS="\t"}
{
    if ($2 < $3)
        print $1, $2, $3
    else
        print $1, $3, $2
}
' "$GT_TXT" > GT.all.bed

if [[ "$CHR" != "ALL" ]]; then
    awk -v chr_list="$CHR" '
    BEGIN{
        OFS="\t"
        n = split(chr_list, a, ",")
        for (i = 1; i <= n; i++) {
            chr[a[i]] = 1
        }
    }
    $1 in chr
    ' GT.all.bed > GT.use.bed
else
    cp GT.all.bed GT.use.bed
fi

########################################
# GT total bp
########################################
GT_TOTAL_BP=$(awk '{sum += $3-$2} END{print sum+0}' GT.use.bed)

########################################
# Loop over models
########################################
for MODEL_TXT in $MODELS; do
    NAME=$(basename "$MODEL_TXT" .txt)

    ####################################
    # Prepare MODEL BED
    ####################################
    awk '
    BEGIN{OFS="\t"}
    {
        if ($2 < $3)
            print $1, $2, $3
        else
            print $1, $3, $2
    }
    ' "$MODEL_TXT" > MODEL.all.bed

    if [[ "$CHR" != "ALL" ]]; then
        awk -v chr_list="$CHR" '
        BEGIN{
            OFS="\t"
            n = split(chr_list, a, ",")
            for (i = 1; i <= n; i++) {
                chr[a[i]] = 1
            }
        }
        $1 in chr
        ' MODEL.all.bed > MODEL.use.bed
    else
        cp MODEL.all.bed MODEL.use.bed
    fi


    ####################################
    # Recall: GT covered by MODEL
    ####################################
    HIT_GT_BP=$(bedtools intersect \
      -a GT.use.bed \
      -b MODEL.use.bed \
      -wo \
    | awk -v fgt="$FRAC_GT" -v fmd="$FRAC_MODEL" '
    {
      gt_id = $1":"$2"-"$3
      gt_len = $3 - $2
      md_len = $6 - $5
      ov = $NF

      if (ov/gt_len >= fgt && ov/md_len >= fmd) {
          if (ov > best[gt_id])
              best[gt_id] = ov
          len[gt_id] = gt_len
      }
    }
    END {
      sum = 0
      for (id in best) {
          if (best[id] > len[id])
              sum += len[id]
          else
              sum += best[id]
      }
      print sum+0
    }')



    RECALL=$(awk -v h="$HIT_GT_BP" -v t="$GT_TOTAL_BP" \
        'BEGIN{if(t>0) printf "%.6f", h/t; else print 0}')

    ####################################
    # Precision: MODEL covered by GT
    ####################################
    MODEL_TOTAL_BP=$(awk '{sum += $3-$2} END{print sum+0}' MODEL.use.bed)

    HIT_MODEL_BP=$(bedtools intersect \
    -a MODEL.use.bed \
    -b GT.use.bed \
    -wo \
    | awk -v fgt="$FRAC_GT" -v fmd="$FRAC_MODEL" '
    {
        md_id = $1":"$2"-"$3
        md_len = $3 - $2
        gt_len = $6 - $5
        ov = $NF

        if (ov/md_len >= fmd && ov/gt_len >= fgt) {
            if (ov > best[md_id])
                best[md_id] = ov
            len[md_id] = md_len
        }
    }
    END {
        sum = 0
        for (id in best) {
            if (best[id] > len[id])
                sum += len[id]
            else
                sum += best[id]
        }
        print sum+0
    }')


    PRECISION=$(awk -v h="$HIT_MODEL_BP" -v t="$MODEL_TOTAL_BP" \
        'BEGIN{if(t>0) printf "%.6f", h/t; else print 0}')

    ####################################
    # Output
    ####################################
    echo -e "${NAME}\t${RECALL}\t${PRECISION}" >> "$OUT"

    rm -f MODEL.all.bed MODEL.use.bed
done

rm -f GT.all.bed GT.use.bed

echo "[DONE] TAD evaluation finished → $OUT"
