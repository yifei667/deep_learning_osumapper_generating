:: Modify name fields before running
npm i polynomial
python map_reader.py [Osu File Path] [ffmpeg Path]
pause
python rhythm_evaluator.py [Osu File Path] [Model_1] [Model_2] (Optional: Mapthis Npz File)
pause
python GAN.py rhythm_data.npz flow_dataset.npz
pause