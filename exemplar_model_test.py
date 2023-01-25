import numpy as np
import matplotlib.pyplot as plt

# To-Do
# how is exemplar space seeded?
# population vectors for similarity bias?

def mink_r_metric(i,j,r=2):
    '''
    Minkowski r-metric, see "Attention, Similarity, and
    the Identification-Categorization Relationship" (p.2, Eq 3, Nosofsky, 1986)

    Parameters
    ----------
    i : np.array (2,)
        exemplar i.
    j : np.array (2,)
        exemplar j.
    r : int
        1: city block metric, 2: euclidian distance.

    Returns
    -------
    float
        distance score.

    '''
    assert i.shape == (2,)
    assert j.shape == (2,)
    return np.sum(np.absolute(i-j))**1/r


def eta_dist(i,j,k=1):
    '''
    Eta distance, see "Attention, Similarity, and
    the Identification-Categorization Relationship" (p.2, Eq 4a&b, Nosofsky, 1986).
    
    In "Lexical contrast maintenance and the organization of sublexical contrast systems" (Wedel,2012)
    there seems to be an extension with a scaling factor k set to 0.2.

    Parameters
    ----------
    i : np.array (2,)
        exemplar i.
    j : np.array (2,)
        exemplar j.
    k : float
        scaling factor from Wedel(2012), appendix,p.36, Eq 1.

    Returns
    -------
    float
        distance score.

    '''
    assert i.shape == (2,)
    assert j.shape == (2,)
    return np.exp(-k*mink_r_metric(i,j))


def word_cat_sim(i,word_cat):
    '''
    Numerator Eq 5, p.2, Nosokfsky(1986).
    
    MISSING: bias b_J from Nosofsky (1986).

    Parameters
    ----------
    i : np.array (2,)
        target exemplar i.
    word_cat : np.array (100,2)
        array with all exemplars of word category.

    Returns
    -------
    float
        summed similarity between target exemplar and word cat.

    '''
    # Ignore init exemplars [0,0]
    # As list
    #J = [list(j) for j in word_cat if j[0]!=0 and j[1]!=0]
    
    # As array
    J_flat = word_cat[word_cat != 0]
    J = np.reshape(J_flat,(-1,2))
    
    return np.sum(np.array([eta_dist(i,j) for j in J]))


def insert_word(space,word_cat,word):
    '''
    Insert word in specified word_category in exemplar space.
    Insert at first position, delete last.
    '''
    new_wc = np.insert(space[word_cat][0:-1],0,word) # create new word cat array, shape (200,)
    new_wc = np.reshape(new_wc,(100,-1)) # reshape as (100,2)
    space[word_cat] = new_wc
    return space

class Agent:
    def __init__(self,name):
        
        self.name = name
        
        #### Initialize space ####
        self.space = np.zeros((4,100,2))
        
        #### Initialize word categories ####
        # init word cat 1 == 'ba'
        #self.space = insert_word(self.space, 0, [26,26])
        #self.space = insert_word(self.space, 0, [25,25])
        self.space = insert_word(self.space, 0, [2,2])
        self.space = insert_word(self.space, 0, [1,1])
        
        # init word cat 2 == 'pa'
        #self.space = insert_word(self.space, 1, [76,26])
        #self.space = insert_word(self.space, 1, [75,25])
        self.space = insert_word(self.space, 1, [99,2])
        self.space = insert_word(self.space, 1, [98,1])
        
        # init word cat 3 == 'bi'
        #self.space = insert_word(self.space, 2, [26,76])
        #self.space = insert_word(self.space, 2, [25,75])
        self.space = insert_word(self.space, 2, [2,99])
        self.space = insert_word(self.space, 2, [1,98])
        
        # init word cat 4 == 'pi'
        #self.space = insert_word(self.space, 3, [76,76])
        #self.space = insert_word(self.space, 3, [75,75])
        self.space = insert_word(self.space, 3, [99,99])
        self.space = insert_word(self.space, 3, [98,98])
        
        self.words = np.reshape(self.space,(self.space.shape[0]*self.space.shape[1],-1))
        self.segments_dim1 = self.space[:,:,0].flatten()
        self.segments_dim2 = self.space[:,:,1].flatten()
        
        # self.space:
        # axis 0 = word cats
        # axis 1 = words in each cat
        # axis 2 = segments
        
        #### Activations ####
        positions = np.arange(1,101,1,dtype=int)
        self.activations = np.exp(-0.2*positions)
        
    def choose_rword(self,wc_index):
        #wc_index = random.randint(0, 3)# randomly select word cat index
        #print('Produced cat: ',wc_index)
        word_cat = self.space[wc_index] # select word cat to pick word from
        choice_probs = self.activations/self.activations.sum() # turning activations into prob distr
        random_word_idx = np.random.choice(len(word_cat),1,p=choice_probs) # sample index given prob distr
        random_word = word_cat[random_word_idx] # pick word from chosen word cat with chosen index
        #print('random word: ', random_word)
        
        # Ensure that no exemplar [0,0] is picked
        if random_word.all()!=0:
            #print('not all 0')
            return random_word.flatten()
        else:
            #print('all 0')
            return self.choose_rword(wc_index)
        
    def p_vec(self,word,wc_index,k = 0.2):
        '''
        Attempt to implement population vector formula from Wedel(2012), appendix, p.36, Eq 1.

        Parameters
        ----------
        word : np.array (2,)
            Production target x.
        wc_index : int
            word category index.

        Returns
        -------
        p : np.array (2,)
            population vector.

        '''
        p_num = np.array([0,0])
        p_den = np.array([0,0])
        x = word
        Y = self.space[wc_index]
        Y_flat = Y[Y != 0] # filtering out init exemplars [0,0]
        Y = np.reshape(Y_flat,(-1,2))
        a_ind = 0
        for y in Y:
            #a_ind = np.where(Y==y)[0][0] # we want only the first row index of np.where
            # print('a_ind: ',a_ind) # potential problem: this gives only index of first match (see zero example)
            
            w = self.activations[a_ind]
            p_num = p_num+(y*w*np.exp(-k*np.abs(x-y)))
            p_den = p_den+(w*np.exp(-k*np.abs(x-y)))
            #print('p_num: ',p_num)
            #print('p_den: ',p_den)
            a_ind+=1
        p = p_num/p_den
        return p
            
                
    def produce(self,wc_index):
        prod_target = self.choose_rword(wc_index)
        p_vector = self.p_vec(prod_target,wc_index)
        noise = np.random.normal(0,3,1) # Random noise, mean 0 std 3, used during output creation
        noise = np.absolute(noise)
        p_noise = np.where(p_vector<50,p_vector+noise,p_vector-noise) # if smaller 50, add noise, else substract
        return p_noise
        
    def receive(self,word):
        cat_sims = np.array([word_cat_sim(word, wc) for wc in self.space])
        cs_sum = np.sum(cat_sims) # Nosofsky(1986),p.2,Eq 5, denominator
        wc_probs = np.array([cs/cs_sum for cs in cat_sims])
        max_cat = np.argmax(wc_probs)
        self.space = insert_word(self.space, max_cat, word)
                
    def plot_space(self):
        x = self.space[:,:,0].flatten()
        x = x[x!=0] # remove 0s for plotting
        y = self.space[:,:,1].flatten()
        y= y[y!=0] # remove 0s for plotting
        plt.scatter(x,y)
        plt.title(self.name)
        plt.show()
        
        
def pp_loop(iterations,agent1,agent2):
    for i in range(iterations):
        # Agent 1 produces, agent 2 receives
        agent2.receive(agent1.produce(0))
        agent2.receive(agent1.produce(1))
        agent2.receive(agent1.produce(2))
        agent2.receive(agent1.produce(3))
        
        # Agent 2 produces, agent 1 receives
        agent1.receive(agent2.produce(0))
        agent1.receive(agent2.produce(1))
        agent1.receive(agent2.produce(2))
        agent1.receive(agent2.produce(3))
        
        agent2.plot_space()
    print(agent1.space)
    
    
        
if __name__=='__main__':
    agent1 = Agent('Agent 1')
    agent2 = Agent('Agent 2')
    pp_loop(100,agent1,agent2)

