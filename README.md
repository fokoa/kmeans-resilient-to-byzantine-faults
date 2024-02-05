# kmeans-resilient-to-byzantine-faults

## Use

After downloading this repository and installing the libraries contained in `requirements.txt`, go to the `code` folder and run one of the `main_*.py` files as follows:

`mpiexec -n 11 python main_iris.py -b 5`

or

`mpiexec -n 11 python main_iris.py --byzantine 5`


* `11`: stand for total number of machines in the system (10 workers et 1 coordinator)
* `5`: stand for total number of byzantine workers (it must be at most half of the total number of machines)

