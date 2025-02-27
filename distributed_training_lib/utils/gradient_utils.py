import numpy as np
from mpi4py import MPI
from typing import List, Union, Dict

class GradientManager:
    """Class to manage gradient operations in distributed training."""
    
    @staticmethod
    def average_gradients(gradients: np.ndarray, dp_comm: MPI.Comm) -> np.ndarray:
        """Average gradients across data parallel processes.
        
        Args:
            gradients: Local gradients from the current process
            dp_comm: Data parallel communicator
            
        Returns:
            Averaged gradients
        """
        # Sum gradients across all data parallel processes
        summed_gradients = np.zeros_like(gradients)
        dp_comm.Allreduce(gradients, summed_gradients, op=MPI.SUM)
        
        # Average by dividing by the number of processes
        dp_size = dp_comm.Get_size()
        averaged_gradients = summed_gradients / dp_size
        
        return averaged_gradients
    
    @staticmethod
    def collect_gradients(gradients: np.ndarray, dp_comm: MPI.Comm) -> List[np.ndarray]:
        """Collect gradients from all data parallel processes.
        
        Args:
            gradients: Local gradients from the current process
            dp_comm: Data parallel communicator
            
        Returns:
            List of gradients from all processes
        """
        return dp_comm.allgather(gradients)
    
    @staticmethod
    def reduce_gradients(gradients_dict: Dict[str, np.ndarray], dp_comm: MPI.Comm, 
                         average: bool = True) -> Dict[str, np.ndarray]:
        """Reduce multiple gradients across data parallel processes.
        
        Args:
            gradients_dict: Dictionary of local gradients {name: gradient}
            dp_comm: Data parallel communicator
            average: Whether to average gradients after reduction
            
        Returns:
            Dictionary of reduced gradients
        """
        # Create a dictionary to hold the reduced gradients
        reduced_gradients = {}
        
        # Reduce each gradient separately
        for name, grad in gradients_dict.items():
            reduced_grad = np.zeros_like(grad)
            dp_comm.Allreduce(grad, reduced_grad, op=MPI.SUM)
            
            if average:
                dp_size = dp_comm.Get_size()
                reduced_grad = reduced_grad / dp_size
                
            reduced_gradients[name] = reduced_grad
            
        return reduced_gradients 