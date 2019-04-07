EPOCHS=1000
BATCH_SIZE=128
SAVEDIR=2019\_01\_10

echo START
echo `date '+%y/%m/%d %H:%M:%S'`
##regression
#for i in `seq 0 2`
#do
#python regression/chainer_regression.py delaney.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ECFP_delaney_$i.txt
#done
#echo Exit ECFP_delaney
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression/chainer_regression.py malaria.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ECFP_malaria_$i.txt
#echo Exit ECFP_malaria $i
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression/chainer_regression.py     cep.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ECFP_cep.txt
#echo Exit ECFP_cep
#echo `date '+%y/%m/%d %H:%M:%S'`
#
##regression_1_dim
#python regression_1_dim/chainer_regression.py delaney.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/1_dim_delaney.txt
#echo Exit 1_dim_delaney
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_1_dim/chainer_regression.py malaria.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/1_dim_malaria.txt
#echo Exit 1_dim_malaria
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_1_dim/chainer_regression.py     cep.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/1_dim_cep.txt
#echo Exit 1_dim_cep
#echo `date '+%y/%m/%d %H:%M:%S'`
#
##regression_fcfp
#python regression_fcfp/chainer_regression.py delaney.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/FCFP_delaney.txt
#echo Exit FCFP_delaney
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_fcfp/chainer_regression.py malaria.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/FCFP_malaria_$i.txt 
#echo Exit FCFP_malaria $i
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_fcfp/chainer_regression.py     cep.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/FCFP_cep.txt     
#echo Exit FCFP_cep
#echo `date '+%y/%m/%d %H:%M:%S'`
#
##regression_ecfc
#python regression_ecfc/chainer_regression.py delaney.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/CONCAT_delaney.txt
#echo Exit CONCAT_delaney
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_ecfc/chainer_regression.py malaria.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/CONCAT_malaria.txt  
#echo Exit CONCAT_malaria
#echo `date '+%y/%m/%d %H:%M:%S'`
#python regression_ecfc/chainer_regression.py     cep.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/CONCAT_cep.txt      
#echo Exit CONCAT_cep
#echo `date '+%y/%m/%d %H:%M:%S'`

#regression_attention
#python regression_attention/chainer_regression.py delaney.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ATTENTION_delaney.txt
#echo Exit ATTENTION_delaney
echo `date '+%y/%m/%d %H:%M:%S'`
python regression_attention/chainer_regression.py malaria.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ATTENTION_malaria.txt  
echo Exit ATTENTION_malaria
echo `date '+%y/%m/%d %H:%M:%S'`
python regression_attention/chainer_regression.py     cep.csv --epochs $EPOCHS --batch_size $BATCH_SIZE > $SAVEDIR/ATTENTION_cep.txt      
echo Exit ATTENTION_cep
echo `date '+%y/%m/%d %H:%M:%S'`

