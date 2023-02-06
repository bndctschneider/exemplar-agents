import numpy as np
import matplotlib.pyplot as plt

def mink_r_metric(i,j,r=1):
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
    return np.sum(np.absolute(i-j)**r)**1/r


def eta_dist(i,j,k=0.2):
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



def insert_word(space,word_cat,word):
    '''
    Insert word in specified word_category in exemplar space.
    Insert at first position, delete last.
    
    Parameters
    ----------
    space : np.array (4,100,2)
        exemplar space.
    word_cat : np.array (100,2)
        array with all exemplars of word category.

    Returns
    -------
    np.array (4,100,2)
        new space with inserted word exemplar.
    '''
    new_wc = np.insert(space[word_cat][0:-1],0,word) # create new word cat array, shape (200,)
    new_wc = np.reshape(new_wc,(100,-1)) # reshape as (100,2)
    space[word_cat] = new_wc
    return space


def rand_exemplar(i):
    '''
    Given a reference exemplar i, it adds some random noise.
    Used for seeding the model.

    Parameters
    ----------
    i : np.array (2,)
        reference exemplar.

    Returns
    -------
    exemplar close to reference exemplar.

    '''
    
    assert i.shape == (2,)
    
    rvals = np.random.normal(scale=3.0, size=(2,))
    
    return i+rvals
    

class Agent:
    def __init__(self,name):
        
        self.name = name
        
        #### Initialize space ####
        self.space = np.zeros((4,100,2))
        
        #### Initialize word categories ####
        # init word cat 0 == 'ba'
        wc0_ref = np.array([25,25])
        for i in range (100):
            self.space = insert_word(self.space, 0, rand_exemplar(wc0_ref))
        
        # init word cat 1 == 'pa'
        wc1_ref = np.array([75,25])
        for i in range (100):
            self.space = insert_word(self.space, 1, rand_exemplar(wc1_ref))
        
        # init word cat 2 == 'bi'
        wc2_ref = np.array([25,75])
        for i in range (100):
            self.space = insert_word(self.space, 2, rand_exemplar(wc2_ref))
        
        # init word cat 3 == 'pi'
        wc3_ref = np.array([75,75])
        for i in range (100):
            self.space = insert_word(self.space, 3, rand_exemplar(wc3_ref))
        
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
        self.activation_space = np.array([self.activations,self.activations,self.activations,self.activations])
        
    def get_activation(self,word):
        '''
        Parameters
        ----------
        word : np.array (2,)
            word exemplar for which we want to retrieve activation.

        Returns
        -------
        a: float
            activation value.

        '''
        word_loc = np.where(self.space==word)
        ind1 = word_loc[0][0]
        ind2 = word_loc[1][0]
        return self.activation_space[ind1][ind2]
        
        
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
        
    
    def p_vec_w(self,word,wc_index,k = 0.2):
        '''
        Population vector formula from Wedel(2012), appendix, p.36, Eq 1.
        Word level.

        Parameters
        ----------
        word : np.array (2,)
            Production target x.
        wc_index : int
            word category index.

        Returns
        -------
        p : np.array (2,)
            word level population vector.

        '''
        p_num = np.array([0,0])
        p_den = np.array([0,0])
        x = word
        Y = self.space[wc_index]
        Y_flat = Y[Y != 0] # filtering out init exemplars [0,0]
        Y = np.reshape(Y_flat,(-1,2))
        for y in Y:
            
            w = self.get_activation(y)
            p_num = p_num+(y*w*np.exp(-k*np.abs(x-y)))
            p_den = p_den+(w*np.exp(-k*np.abs(x-y)))
            
            
        p = p_num/p_den
        return p
    
    
    def p_vec_s(self,word,k = 0.2):
        '''
        Population vector formula from Wedel(2012), appendix, p.36, Eq 1.
        Segment level.

        Parameters
        ----------
        word : np.array (2,)
            Production target x.
        

        Returns
        -------
        p : np.array (2,)
            segment level population vector.

        '''
        p_num = np.array([0,0])
        p_den = np.array([0,0])
        x = word
        for wc in self.space:
            Y = wc
            Y_flat = Y[Y != 0] # filtering out init exemplars [0,0]
            Y = np.reshape(Y_flat,(-1,2))
            for y in Y:
                
                w = self.get_activation(y)
                p_num = p_num+(y*w*np.exp(-k*np.abs(x-y)))
                p_den = p_den+(w*np.exp(-k*np.abs(x-y)))
    
        p = p_num/p_den
        return p
    
    def default_b(self,p_vec):
        '''
        Our default for adding noise on the population_vector.

        Parameters
        ----------
        p_vec : np.array (2,)
            combined population vector.

        Returns
        -------
        population vector with added noise towards center.

        '''
        noise = np.random.normal(0,3,1) # Random noise, mean 0 std 3, used during output creation
        noise = np.absolute(noise)
        p_noise = np.where(p_vec<50,p_vec+noise,p_vec-noise) # if smaller 50, add noise, else substract
        return p_noise
        
    def wedel_b(self,p_vec,N=100,G=5000):
        '''
        Bias (noise) formula from Wedel(2012), appendix, p.36, Eq 2.

        Parameters
        ----------
        p_vec : np.array (2,)
            combined population vector.
        N : int
            number of points in the space. The default is 100.

        Returns
        -------
        population vector with added noise towards center.

        '''
        b = (p_vec-N/2)**2/G
        p_noise = np.where(p_vec<50,p_vec+b,p_vec-b) # if smaller 50, add noise, else substract
        return p_noise
    
    def word_cat_sim(self,i,word_cat):
        '''
        Numerator Eq 5, p.2, Nosokfsky(1986).
        
        MISSING: bias b_J from Nosofsky (1986).
        
        ADDITION: multiply by activation value from Wedel(2012)

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
        a = np.array([self.get_activation(j) for j in J])
        
        return np.sum(a*np.array([eta_dist(i,j) for j in J]))
                
    def produce(self,wc_index):
        prod_target = self.choose_rword(wc_index)
        p_vector = (9*self.p_vec_w(prod_target,wc_index)+self.p_vec_s(prod_target))/10
        #print('p_vec_w: ',self.p_vec_w(prod_target, wc_index))
        #print('p_vec_s: ',self.p_vec_s(prod_target))
        #print('p_vector: ',p_vector)
        
        #p_noise = self.default_b(p_vector)
        p_noise = self.wedel_b(p_vector)
        #p_noise = p_vector
        return p_noise
    
    def aab_store1(self,max_sim):
        r = np.random.random()
        store = True if r<max_sim else False
        return store
    
    def aab_store2(self,wc_probs): # this doesn't work because of 'underflow'
        store_values = np.array([0,1]) # don't store or store
        # probability of not being stored:
        #r_max = (1/max(cat_sims))/cs_sum # reciprocal of max cat sim divided by sum of sim to all cats
        #print('r_max:',r_max)
        store_probs = np.array([1-np.max(wc_probs),np.max(wc_probs)])
        store_val = np.random.choice(store_values,p=store_probs)
        store = True if store_val==1 else False
        return store
        
    def receive(self,word,aab=True):
        cat_sims = np.array([self.word_cat_sim(word, wc) for wc in self.space])
        max_cs = np.max(cat_sims)
        
        cs_sum = np.sum(cat_sims) # Nosofsky(1986),p.2,Eq 5, denominator
        
        # Uncomment to see 'underflow' problem:
        # max_cs/cs_sum will always be 1, because all other sims than the max one are so tiny
        # that they aren't added on top of the max one in cs_sum,
        # therefore max_cs == cs_sum
        #print('cat_sims:',cat_sims)
        #print('max cat sim: ',max_cs)
        #print('cs_sum: ',cs_sum)
        #print('max_cs/cs_sum',max_cs/cs_sum)
        
        wc_probs = np.array([cs/cs_sum for cs in cat_sims])
        max_cat = np.argmax(wc_probs)
        max_cat_prob = np.max(wc_probs)
        
        # Anti ambiguity bias
        if aab:
            store= self.aab_store1(max_cat_prob)
            #store= self.aab_store2(wc_probs)
            if store:
                self.space = insert_word(self.space, max_cat, word)
            else:
                print('not stored')
        else:
            self.space = insert_word(self.space, max_cat, word)
                
    def plot_space(self,act_alpha=False): 
        fig, ax = plt.subplots()
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        z = self.activations
        colors = ['r','g','b','y']
        markers = ["v" , "o" , "," , "x"]
        for i in range(4):
            x = self.space[i,:,0].flatten()
            x = x[x!=0] # remove 0s for plotting
            y = self.space[i,:,1].flatten()
            y= y[y!=0] # remove 0s for plotting
            if act_alpha:
                ax.scatter(x,y, color=colors[i],marker=markers[i],alpha=z)
            else:
                ax.scatter(x,y, color=colors[i],marker=markers[i])
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
    #print(agent1.space)
    
    
        
if __name__=='__main__':
    agent1 = Agent('Agent 1')
    agent2 = Agent('Agent 2')
    pp_loop(150,agent1,agent2)

