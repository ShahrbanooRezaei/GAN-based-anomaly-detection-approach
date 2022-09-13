conda create -n MyGanomaly python=3.7
conda activate MyGanomaly
conda install -c intel mkl_fft
conda install -c pytorch pytorch (1.7.1)
conda install -c pytorch torchvision (0.8.2)
conda install -c conda-forge torchfile (0.1.0)
conda install tornado=6.1
conda install -c conda-forge tqdm (4.54.1)
conda install urllib3==1.26.2
conda install -c conda-forge visdom (0.1.8.9)
conda install websocket-client==0.56.0

cd 
pip install --user --requirement requirements.txt

conda install spyder
conda install pandas
