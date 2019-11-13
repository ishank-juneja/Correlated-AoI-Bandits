instances=( '../instances/i-1.txt' '../instances/i-2.txt' '../instances/i-3.txt' )

hz=204800

op='outputData.txt'
echo begin

for i in "${instances[@]}";
do
    python mutli_armed_bandit.py --instance $i --algorithm $al --randomSeed $rs --epsilon $eps --horizon $hz >> $op

done;
