# caltech-ee148-spring2020-hw03

The link to the instructions is available on Piazza.

To train the model, run:
  python main.py --batch-size 32 --epochs 10

To generate the plot of training and validadtion loss versus the number of training examples, run:
  python main.py --batch-size 32 --epochs 10 --test-datasize
  
To evaluate the model, run:
  python main.py --evaluate --load-model your_model_file.pt
