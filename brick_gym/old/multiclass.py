import multiprocessing

class UnknownMultiClassCommand(Exception):
    pass

def remote_process(connection, constructor, constructor_args):
    instance = constructor(**constructor_args)
    
    while True:
        command, *arguments = connection.recv()
        if command == 'SHUTDOWN':
            break
        
        elif command == 'SETATTR':
            attr, value = arguments
            setattr(instance, attr, value)
        
        elif command == 'GETATTR':
            attr = arguments[0]
            value = getattr(instance, attr)
            connection.send(value)
        
        elif command == 'METHOD':
            method, kwargs = arguments
            result = getattr(instance, method)(**kwargs)
            connection.send(result)
        
        else:
            raise UnknownMultiClassCommand(
                    'Unknown multi-class command: %s'%command)
    
class MultiClass:
    def __init__(self, num_processes, constructor, constructor_args):
        self.num_processes = num_processes
        self.connections = []
        self.processes = []
        self.constructor = constructor
        self.constructor_args = constructor_args
    
    def start_processes(self):
        for i in range(self.num_processes):
            parent_connection, child_connection = multiprocessing.Pipe()
            self.connections.append(parent_connection)
            process = multiprocessing.Process(
                    target = remote_process,
                    args = (child_connection,
                            self.constructor,
                            self.constructor_args[i]))
            process.start()
            self.processes.append(process)
    
    def shutdown_processes(self):
        for connection in self.connections:
            connection.send(('SHUTDOWN', None, None))
        
        for process in self.processes:
            process.join()
        
        self.connections = []
        self.processes = []
    
    def call_method(self, method_name, method_args=None, processes=None):
        if processes is None:
            processes = range(self.num_processes)
        if method_args is None:
            method_args = [{} for _ in processes]
        assert len(method_args) == len(processes)
        for process_id, kwargs in zip(processes, method_args):
            connection = self.connections[process_id]
            connection.send(('METHOD', method_name, kwargs))
        
        result = []
        for process_id in processes:
            connection = self.connections[process_id]
            result.append(connection.recv())
        
        return result
    
    def get_attr(self, attr, processes=None):
        if processes is None:
            processes = range(self.num_processes)
        for process_id in processes:
            connection = self.connections[process_id]
            connection.send(('GETATTR', attr))
        
        result = []
        for process_id in processes:
            connection = self.connections[process_id]
            result.append(connection.recv())
        
        return result
    
    def set_attr(self, attr, values, processes=None):
        if processes is None:
            processes = range(self.num_processes)
        for process_id in processes:
            connection = self.connections[process_id]
            connection.send(('SETATTR', attr, values[i]))
    
    def __del__(self):
        self.shutdown_processes()
    
    def __enter__(self):
        self.start_processes()
    
    def __exit__(self, type, value, traceback):
        self.shutdown_processes()
