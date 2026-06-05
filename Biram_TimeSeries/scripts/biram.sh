#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

rm -rf /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls

python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Ethanol_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/EthanolConcentration --data_class tsra --epochs 250 --lr 0.0005 --lr_step 50,150 --lr_factor 0.5 --batch_size 64 --optimizer RAdam --pos_encoding learnable --l2_reg 1e-3 --ktimes 2 --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Ethanol.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name SCP2_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/SelfRegulationSCP2 --data_class tsra --epochs 150 --lr 0.001 --lr_step 50,100 --lr_factor 0.5 --batch_size 64 --optimizer RAdam --pos_encoding learnable --l2_reg 1e-3 --dropout 0.3 --ktimes 2 --activation gelu --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/SCP2.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name SCP1_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/SelfRegulationSCP1 --data_class tsra --epochs 200 --lr 0.0005 --lr_step 50,100,150 --lr_factor 0.5 --batch_size 16 --optimizer RAdam --pos_encoding learnable --ktimes 2 --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/SCP1.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Heartbeat_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/Heartbeat --data_class tsra --epochs 250 --lr 0.0005 --lr_step 50,150 --lr_factor 0.5 --batch_size 32 --optimizer RAdam --pos_encoding fixed --l2_reg 1e-2 --dropout 0.5 --ktimes 1 --activation gelu --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Heartbeat.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name UWave_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/UWaveGestureLibrary --data_class tsra --epochs 300 --lr 0.0005 --lr_step 150,250 --lr_factor 0.5 --batch_size 16 --optimizer Adam --pos_encoding learnable --l2_reg 1e-4 --dropout 0.2 --ktimes 1 --activation gelu --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/UWave.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Handwriting_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/Handwriting --data_class tsra --epochs 300 --lr 0.001 --lr_step 150,250 --lr_factor 0.5 --batch_size 64 --optimizer RAdam --pos_encoding fixed --normalization_layer BatchNorm --ktimes 2 --activation gelu --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Handwriting.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name PEMS_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/PEMS-SF --data_class tsra --epochs 500 --lr 0.0005 --lr_step 100,200,300,400 --lr_factor 0.5 --batch_size 16 --optimizer RAdam --pos_encoding learnable --normalization_layer BatchNorm --ktimes 1 --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/PEMS.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Arabic_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/SpokenArabicDigits --data_class tsra --epochs 250 --lr 0.001 --lr_step 100,180 --lr_factor 0.5 --batch_size 16 --optimizer Adam --pos_encoding fixed --normalization_layer LayerNorm --ktimes 1 --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Arabic.log 2>&1


python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Face_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/FaceDetection --data_class tsra --epochs 150 --lr 0.0001 --lr_step 20,40,80 --lr_factor 0.5 --batch_size 15 --optimizer RAdam --pos_encoding fixed --l2_reg 1e-3 --dropout 0.5 --ktimes 1 --activation relu --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Face.log 2>&1

python /workspace/Bi-RAM/Biram_TimeSeries/main.py --model biram --output_dir /workspace/TimeSeries_save --name Vowels_BiRAM --records_file /workspace/Bi-RAM/Biram_TimeSeries/results/Classification_records.xls --data_dir /workspace/TimeSeries/JapaneseVowels --data_class tsra --epochs 250 --lr 0.001 --batch_size 32 --optimizer Adam --pos_encoding fixed --ktimes 1 --seed 2026 --task classification --key_metric accuracy > /workspace/Bi-RAM/Biram_TimeSeries/out_log/Vowels.log 2>&1


chmod -R 777 /workspace/Bi-RAM/Biram_TimeSeries/results/
chmod -R 777 /workspace/Bi-RAM/Biram_TimeSeries/out_log/
echo "Final LR Surgery applied. Computing theoretical limits!"