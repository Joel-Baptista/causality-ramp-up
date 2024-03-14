import timeit

def time_func(func, kargs={}):
    start_time = timeit.default_timer()
    res = func(**kargs)
    elapsed_time = timeit.default_timer() - start_time
    print("Execution time of", func.__name__, ":", elapsed_time, "seconds")
    
    return res
