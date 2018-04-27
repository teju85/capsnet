#!/bin/bash
epochs=25

for dataset in mnist cifar10; do
    for detach in 0 1; do
        log=${dataset}_d${detach}.log
        arg=""
        if [ "$detach" = "0" ]; then
            arg="-no-detach $arg"
        fi
        cmd="python net.py -epoch $epochs -root $dataset $arg"
        echo "Running: $cmd"
        if [ -e $log ]; then
            echo "Skipping it. Logfile already exists"
            continue
        fi
        $cmd 2>&1 | tee $log
        echo "Done"
    done
done

echo "| Dataset  | detach? | Train acc | Train time | Test acc | Test time |"
echo "|----------|---------|-----------|------------|----------|-----------|"
for dataset in mnist cifar10; do
    for detach in 0 1; do
        log=${dataset}_d${detach}.log
        if [ "$detach" = "0" ]; then
            d_=" no      "
        else
            d_=" yes     "
        fi
        trainacc=`grep "Train " $log  | tail -n1 | awk '{print $5}' | sed -e 's/.*://'`
        traintime=`grep "Train " $log  | tail -n1 | awk '{print $3}' | sed -e 's/.*://'`
        testacc=`grep "Test " $log  | tail -n1 | awk '{print $5}' | sed -e 's/.*://'`
        testtime=`grep "Test " $log  | tail -n1 | awk '{print $3}' | sed -e 's/.*://'`
        printf "| %8s | %7s | %9s | %10s | %8s | %9s |\n" \
               $dataset $d_ $trainacc $traintime $testacc $testtime
    done
done
