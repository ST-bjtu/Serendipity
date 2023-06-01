False=0
True=1
train=${False}
distance=${True}
preTrainEmbed=${False}
hisLen="50"
algos="SASRec"
device="cuda:1"
dataset='ml-1m'
loss="BPR"
random="2022"
l2='0'
epochs="10"
startEpoch="0"
batchSize="64"
learningRate="1e-3"
inW="0.001"
listW="0.1"
trade="0.99"
bestEpoch="25"
discrepencyloss="L2"
disW="0."

pythonPath="/home/teng_shi/anaconda3/bin/python"
runFile="main.py"

commandArg="--algos ${algos} --loss ${loss} --learningRate ${learningRate} --epochs ${epochs} --batchSize ${batchSize} --device ${device}  --discrepencyloss ${discrepencyloss} --dataset ${dataset}  --disW ${disW} --inW ${inW} --listW ${listW} --bestEpoch ${bestEpoch} --startEpoch ${startEpoch}"
commandArg="${commandArg} --hisLen ${hisLen} --l2 ${l2} --trade ${trade} --random ${random}"
if [ $train -eq $True ]
then
    commandArg="${commandArg} --train"
fi

printDir="./${dataset}/output/${algos}"
if [ $algos = 'din' -o $algos = 'NRHUB' -o $algos = 'SASRec' -o $algos = 'GRU4Rec' -o $algos = 'PURS' ]
then
    printFile="${printDir}/${loss}_${device}_lr:${learningRate}_l2:${l2}_his:${hisLen}_random:${random}_trade:${trade}"
elif [ $algos = 'SNPR' ]
then 
    printFile="${printDir}/${loss}_${device}_lr:${learningRate}_l2:${l2}_his:${hisLen}_random:${random}"
elif [ $algos = 'din_iv' ]
then    
    printFile="${printDir}/${loss}_${device}_lr:${learningRate}_l2:${l2}_his:${hisLen}_random:${random}"
else
    printFile="${printDir}/${loss}_${device}_lr:${learningRate}_l2:${l2}_disW:${disW}_inW:${inW}_his:${hisLen}_random:${random}_trade:${trade}"
fi

if [ $loss = 'trade' ]
then
    printFile="${printFile}_trade:${trade}"
fi

if [ $distance -eq $True ]
then
    commandArg="${commandArg} --distance"
    printFile="${printFile}_dis"
fi

if [ $loss = "listBPR" ]
then
    printFile="${printFile}_listW:${listW}"
fi

if [ $preTrainEmbed -eq $True ]
then
    commandArg="${commandArg} --preTrainEmbed"
    printFile="${printFile}_preTrain"
fi

printFile="${printFile}.txt"

command="${pythonPath} -u ${runFile} ${commandArg}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo $printFile
echo $commandArg

nohup $command > ${printFile} 2>&1 &

