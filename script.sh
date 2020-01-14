# Modify name fields before running
OSU = [Osu File Path]
FFMPEG = [ffmpeg Path]
MODEL1 = [Model 1]
MODEL2 = [Model 2]
RHYTHM = rhythm_data.npz
FLOW = flow_dataset.npz

npm i polynomial
python map_reader.py $OSU $FFMPEG
read -n 1 -s -r -p "Press any key to continue"

# Optional Fourth Argument: Mapthis Npz File
python rhythm_evaluator.py $OSU $MODEL1 $MODEL2 
read -n 1 -s -r -p "Press any key to continue"

python GAN.py $RHYTHM $FLOW