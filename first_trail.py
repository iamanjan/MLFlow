import mlflow

def calculate_sum(x,y):
    return x+y



if __name__=='__main__':
    #z=calculate_sum(10,20)
    #print(f'the sum is:{z}')
    with mlflow.start_run():
        x,y=10,30
        z=calculate_sum(x,y)
        #tracking the experiment vth mlflow
        mlflow.log_param('x',x)
        mlflow.log_param('y',y)
        mlflow.log_metric('z',z)
