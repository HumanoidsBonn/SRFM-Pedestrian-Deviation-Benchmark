from gibson2.metrics.metrics import Metrics
import numpy as np

class Frechet_distance(Metrics):

    def __init__(self):
        super().__init__()
        self.unit= "m"
        
    def calculate(self, trajectory_a=None, trajectory_b=None, **kwargs):
        '''
        Calculate_frechet_dist: 
        
        This function takes two trajectories to be compared as the input argument and it computes the similarity
        between curves using frechet distance measure. The dynamic programming approach is used to calculate the 
        frechet distance between two trajectories. The explanation of frechet distance can be found in following 
        links:

        1. https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance

        2. Eiter, T., & Mannila, H. (1994). Computing discrete Fréchet distance.
           [http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf]

        3. https://muthu.co/computing-the-discrete-frechet-distance-using-dynamic-programming/
        
        The implemented algorithm is based on [2].

        Parameters:
        
            trajectory_a: numpy.ndarray
                Trajectory information of the curve/path

            trajectory_b: numpy.ndarray
                Trajectory information of the curve/path
        
        Returns:
        
            self.distance_measure: numpy.float64
                Frechet_distance between the given trajectories
        '''
        poses_a = self.get_correct_poses_from_trajectory(trajectory_a, kwargs["config_a"])
        poses_b = self.get_correct_poses_from_trajectory(trajectory_b, kwargs["config_b"])

        # Drop the Z column, if its provided in the trajectory information
        
        if(len(poses_a[0])==3): # checking whether the Z column present in the poses_a
            poses_a = poses_a[:,:2]
        if(len(poses_b[0])==3):
            poses_b = poses_b[:,:2]
        
        # Distance matrix to store the euclidean distance between 2 points in the trajectories
        distance_matrix = np.ones((len(poses_a),len(poses_b)), dtype=np.float64)*-1

        distance_matrix[0][0] = np.linalg.norm(poses_a[0]-poses_b[0])
        for idx_a in range(0, len(poses_a)):
            
            for idx_b in range(0, len(poses_b)):
                
                # Finding euclidean distance between points
                euclidean_dist =  np.linalg.norm( poses_a[idx_a] - poses_b[idx_b] )
                
                if(idx_a==0 and idx_b==0):
                    
                    distance_matrix[idx_a][idx_b]= euclidean_dist
                
                elif(idx_a>0 and idx_b==0):
                    
                    distance_matrix[idx_a][idx_b] = max(distance_matrix[idx_a-1][idx_b], euclidean_dist)
                
                elif(idx_a==0 and idx_b>0):
                    
                    distance_matrix[idx_a][idx_b] = max(distance_matrix[idx_a][idx_b-1], euclidean_dist)
                    
                elif(idx_a>0 and idx_b>0):
                    
                    distance_matrix[idx_a][idx_b] = max (min (distance_matrix[idx_a-1][idx_b],
                                                            distance_matrix[idx_a-1][idx_b-1],
                                                            distance_matrix[idx_a][idx_b-1]) ,
                                                            euclidean_dist)
                    
        self.distance_measure = distance_matrix[-1][-1]
        #
        return self.distance_measure