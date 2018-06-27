# wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
cd Data
python download.py
cd celebA
mkdir training_set
mkdir testing_set
find ./ -name '2*.jpg' -exec mv  --target-directory=testing_set '{}' +;
find ./ -name '*.jpg' -exec mv  --target-directory=training_set '{}' +;
mv testing_set/200000.jpg training_set
cd ..




