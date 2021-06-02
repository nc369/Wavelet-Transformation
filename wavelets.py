

import numpy as np
import cmath as cm



def zeropadding(vector):
    """ 
    adds zero pading to 1 dimensional array to increase its length to a power of 2

    Attributes
    -----------
    vector : ndarray
        takes ndarray of rank 1
        
    Returns
    -------
    ndarray 
        zero padded vector
  
    """
    N = len(vector)
    k = np.ceil( np.log2(N) )
    add = int(2**k - N)
    vec = np.zeros((1,add))
    return np.hstack((vector.reshape((1,-1)),vec))

def e(m, N):
    """
    calculates e^(2pi*i*m/N)

    Attributes
    ----------
    m : int
    N : int

    Returns
    -------
    complex
        value of the expression for m and N
    
    """

    return np.around(cm.exp((2*np.pi*complex(0,1)*m)/N),10)

v_e = np.vectorize(e) #vectorizes e(m,N)

def W(N):
    """
    DFT matrix of size N

    Attributes
    ----------
    N : int
        size of the DFT matrix

    Returns
    -------
    ndarray
        DFT matrix

    """
    return np.around( np.fromfunction(lambda i,j: np.conjugate( v_e(i*j,N) ), (N,N), dtype=complex), 10)

def inv_W(N):
    """
    Inverse DFT matrix of size N

    Attributes
    ----------
    N : int
        size of the DFT matrix

    Returns
    -------
    ndarray
        Inverse DFT matrix

    """
    return np.around(np.conjugate(W(N))/N , 10)

def fft(vector):
    """
    FFT transform of a vector

    Attributes
    ----------
    vector : ndarray of rank 1 or list
        vector to be transformed

    Returns
    -------
    ndarrayof rank 1
        transformed vector

    """

    if type(vector)!=np.ndarray:
        vector=np.array(vector)

    N = vector.shape[0]                         #dimension of vector
    if N % 2==0:

        M=N/2
        fac = np.conjugate(v_e(np.arange(M),N)) # e^(2pi * m/N) for each m in 0,1,...,M-1
        even = fft( vector[0::2] )              # fft of even coordinates
        odd = fft(vector[1::2] )                # fft of odd coordinates
        top = even + fac*odd
        bottom = even - fac*odd
        return np.concatenate((top,bottom))     #computing fft of the original vector

        
    else:
        return  W(N) @ vector

def inv_fft(vector):
    """
    Inverse FFT transform of a vector

    Attributes
    ----------
    vector : ndarray of rank 1 or list
        vector to be transformed

    Returns
    -------
    ndarray of rank 1
        transformed vector

    """
    if type(vector)!=np.ndarray:
        vector=np.array(vector)

    N = vector.shape[0]                         #dimension of vector
    if N % 2==0:
        M=N/2
        fac = v_e(np.arange(M),N)
        even = inv_fft( vector[0::2])           #inv fft of even coord
        odd = inv_fft( vector[1::2] )           #inv fft of odd coord
        top = even + fac*odd
        bottom = even - fac*odd
        return np.concatenate((top,bottom))/2   # inv fft of the vec

        
    else:
        return inv_W(N) @ vector

"""End of setting up FFT and Inverse FFT.
We now construct wavelets for dimension N. However we assume that 2^p | N  where p \in [1,2, ...] 

In fact we can make a weaker assumption that N is even ie the length of the vector to be analysed is even.
"""

def tilda(w):
    """
    tilda operation for a wavelet 
    tilda(v)(n) = conjucate(v(-n))

    Attributes
    ----------
    w : ndarray of rank 1 
        vector to be transformed

    Returns
    -------
    ndarray of rank 1
        transformed vector

    """
    N=w.shape[0]
    v = [np.conjugate(w[-i]) for i in range(N)]
    return np.array(v)

def father_wavelet(N,kind='haar'):
    """
    constructs father wavelet in Z_N of type 'kind'

    Attributes
    ----------
    N : int
        length of father wavelet.
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default 

    Returns
    -------
    ndarray of rank 1
        father wavelet
    
    Raises
    ------
    TypeError
        for any other values of kind

    """
    M=N//2

    if kind=='haar':
        u = np.zeros(N)
        u[0]=u[1]= 1/np.sqrt(2)

    elif kind == 'D6':
        a = 1 - np.sqrt(10)
        b = 1 + np.sqrt(10)
        c = np.sqrt( 5 + 2 * np.sqrt(10) )
        u = np.zeros(N)
        u[0] = b + c
        u[1] = 2*a + 3*b + 3*c
        u[2] = 6*a + 4*b + 2*c
        u[3] = 6*a + 4*b - 2*c
        u[4] = 2*a + 3*b - 3*c
        u[5] = b - c
        u *= np.sqrt(2)/32

    elif kind=='complex':
        u = np.sqrt(2) * np.fromfunction(lambda i,j: np.less(j,M) ,(1,N), dtype=int)[0] #defines u_cap wich has 2 in first half and 0 in remaining
        u = inv_fft(u)

    else:
        raise TypeError("kind has only 3 possible values: 'haar', 'D6' and 'complex'")

    return u

def mother_wavelet(u):
    """
    constructs mother wavelet from father wavelet

    Attributes
    ----------
    u : ndarray of rank 1
        father wavelet.

    Returns
    -------
    ndarray of rank 1
        mother wavelet

    """
    N=u.shape[0]
    new = np.array( [(-1)**(i-1)* np.conj(u[1-i]) for i in range(N)] )  #v(n) = (-1)^(n-1) conj(u[1-n])
    return ( new )

def u_l(u,l):
    """
    constructs u_l from father wavelet u

    Attributes
    ----------
    u : ndarray of rank 1
        father wavelet
    l : int
        level of u_l 

    Returns
    -------
    ndarray of rank 1
        u_l
    

    """
    N=u.shape[0]
    ul = [ np.sum(u[i::N//2**(l-1)]) for i in range(N//2**(l-1))]
    return np.array(ul)

def downsampling(vec,l=1):
    """
    downsampling operator. drops columns

    Attributes
    ----------
    vec : ndarray of rank at most 2
        array to be downsampled
    l : int
        level of downsampling. 1 is default 

    Returns
    -------
    ndarray of rank at most 2
        output is of the same rank as input.
    
    Raises
    ------
    TypeError
        rank of vecor is greater than 2

    """

    if vec.ndim==1:
        N=vec.shape[0]
        new = [ vec[i*2**l]for i in range(N//(2**l))]
        return np.array(new)

    elif vec.ndim==2:
        N=vec.shape[1]
        new = [ vec[:,i*2**l] for i in range(N//(2**l)) ]
        return np.transpose( np.array(new) ) 

    else:
        raise TypeError("vec takes 1 or 2 dimensional ndarray")

def upsampling(vec,l=1):
    """
    upsampling operator. adds columns of 0

    Attributes
    ----------
    vec : ndarray of rank at most 2
        array to be upsampled
    l : int
        level of upsampling. 1 is default 

    Returns
    -------
    ndarray of rank at most 2
        output is of the same rank as input.
    
    Raises
    ------
    TypeError
        rank of vecor is greater than 2

    """

    if vec.ndim==1:
        N=vec.shape[0]
        new = [(vec[i//(2**l)] if i%(2**l)==0  else 0) for i in range(N*2**l) ] 
        return (np.array(new))

    elif vec.ndim==2:
        N=vec.shape[1]
        # adds columns of 0
        new = [ ( vec[:,i//(2**l)] if i%(2**l)==0  else np.zeros(vec.shape[0]) ) for i in range(N*2**l) ] 
        return np.transpose( np.array(new) ) 

    else:
        raise TypeError("vec takes 1 or 2 dimensional ndarray")

def convolution(w,z, matrix=True):

    """
    convolution operator of 2 vectors or 1 vector and an array.

    Attributes
    ----------
    w : ndarray of rank 1
        vector 
    z : ndarray of rank atmost 2
        vector to be convoluted with w. If z is of rank 2, then convolution is 
        performed along axis 0
    matrix : boolean
        to use circulant matrix formed from w as the first column to calculate
        the convolution. default is True

    Returns
    -------
    ndarray of rank at most 2
        output is of the same rank as input.
    
    Raises
    ------
    TypeError
        if matrix is not boolean

    """

    N=w.shape[0]

    if matrix==True:
        w=w[::-1]

        mat=np.empty((N,N),complex)
        #constructs circulant matrix from w as the first column
        for i in range(1,N+1):
            w = np.hstack(( [w[-1]], w[:-1] ))
            mat[i-1]=w
    
        return mat @ z if z.ndim==1 else z @ np.transpose( mat )
        
    
    elif matrix==False:
        ans=[]
        w=w[::-1]
        # defination approach 
        for i in range(1,N+1):
            w = np.hstack(( [w[-1]], w[:-1] ))
            ans.append( z @ w )
    
        return np.transpose( np.array(ans) )
    
    else:
        return TypeError("matrix takes boolean values")

def find_p(N):
    "returns the highest power of 2 that divides int N"
    return 1 + find_p(N//2) if N%2==0 else 0

def p_stage(N,P=-1, kind='D6'):

    """
    constructs p stage wavelets of type 'kind' for Z_N. Note that parameter 
    P != p

    Attributes
    ----------
    N : int
        wavelets for Z_N 
    P : int
        level of the p stage wavelet construction. Takes poitive integer or -1
        as input. If P=-1, constructs p is assumed to be the highest power of 2 
        that divides N. If P > the highest power of 2 that divides N, P is 
        assumed to be -1. Default is -1
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default

    Returns
    -------
    dict
        key values are integers from 1 to p corresponding to the level. Each 
        key corresponds to a tuple of len 2 which returns the father and mother 
        wavelets respectively for that particular key level
    
    Raises
    ------
    TypeError
        if P is not -1 or a positive integer

    """
    
    p=find_p(N)
    if P < -1 or P==0:
        return TypeError("P must be a positive integer or -1")
    elif P==-1 or P>p:
        pass
    else: 
        p=P
    # decides the level p
    wvlts = {}
    
    u = father_wavelet(N,kind)        
    
    v = mother_wavelet(u)
    wvlts[1]=(u,v)

    # construct u_i,v_i for 1<i<=p
    for i in range(2,p+1):
        u_i = u_l(u,i)
        v_i = mother_wavelet(u_i)
        wvlts[i]=(u_i,v_i)
        
    return wvlts

def construct_fg(N, P=-1, kind='haar', matrix=True):

    """
    constructs non iterative filter banks f_1, f_2,...f_p, g_p wavelet basis of 
    type 'kind' for Z_N. 
    Note that parameter P != p

    Attributes
    ----------
    N : int
        wavelets for Z_N 
    P : int
        level of the wavelet construction. Takes poitive integer or -1
        as input. If P=-1, p is assumed to be the highest power of 2 that 
        divides N. If P > the highest power of 2 that divides N, P is assumed to
        be -1. Default is -1
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default
    matrix : boolean
        to use circulant matrix for calculating convolution. default is True

    Returns
    -------
    dict
        key values are integers from 1 to p and 'g'. Any integer key i returns 
        value of ndarray corresponding to f_i. key 'g' corresponds to g_p
    
    Raises
    ------
    TypeError
        if P is not -1 or a positive integer
    
    """

    p=find_p(N)
    if P < -1 or P==0:
        return TypeError("P must be a nongeative integer or 0 ")
    elif P==-1 or P>p:
        pass
    else: 
        p=P
    # decides the level

    book={}

    if kind=='haar':
        temp=np.arange(N)
        #construction of non recursive haar wavelets
        for l in range(1,p+1):
            t = [ 0 if k >= 2**l else (2**((-1*l)/2) if k< 2**(l-1) else -1*2**((-1*l)/2) ) for k in temp ]
            book[l]=np.array(t)            
        l=p
        book['g'] =  np.array( [2**(-l/2) if j< 2**l else 0 for j in temp] )

    elif kind=='complex' or kind=='D6':
        p_stg = p_stage(N,p, kind)
        
        #construction of non recursive complex and D6 wavelets
        for key, pair in p_stg.items():
            
            if key==1:
                f=pair[1]   #pair[1]=v_1
                g=pair[0]   #pair[0]=u_1
            else: 
                f = convolution( upsampling(pair[1],key-1), g , matrix )
                g = convolution( upsampling(pair[0],key-1), g , matrix )
            book[key]=f     #add f_i as i:f in dict book
            if key==p:
                book['g']=g #add g_p as 'g':g in book

    return book

def analysis(z, P=-1, fg=None, kind='haar', matrix=True, compress=False):

    """
    multi stage analysis of vector or vectors z using 'kind' wavelets. 
    returns x_1, x_2, ..., x_p, y_p componenets of z
    Note that parameter P != p

    Attributes
    ----------
    z : ndarray of rank atmost 2
        vector or vectors to be analysed. Incase of rank 2 input, analysis is
        performed along axis 0
    P : int
        level of the wavelet construction. Takes poitive integer or -1
        as input. If P=-1, p is assumed to be the highest power of 2 that 
        divides N. If P > the highest power of 2 that divides N, P is assumed to
        be -1. Default is -1
    fg : dict
        user can provide non iterative filter bank to consruct multiresolution
        analaysis of z. Default is None. If both fg and kind are provided, fg
        takes precedence. refer to function fg_construct() for structure of fg
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default
    matrix : boolean
        to use circulant matrix for calculating convolution. default is True
    compress : boolean
        If True returns only the low pass components else reurns all components
        default is False

    Returns
    -------
    dict
        key values are integers from 1 to p and 'y'. Any integer key i returns 
        value of ndarray corresponding to x_i. key 'y' corresponds to y_p. rank 
        of ndarray values in dict is same as that of the input

    
    """

    N = len(z) if z.ndim==1 else z.shape[1]
    if fg==None:
        fg = construct_fg(N,P,kind,matrix) #construct fg if none provided

    book = {}

    if compress==False:
        for key,pair in fg.items():
            
            if key!='g':
                #constructs x_i
                x = downsampling( convolution( tilda( pair ), z, matrix ) , key)
                book[key] = x

            else:
                #constructs y_p
                y = downsampling( convolution( tilda( pair ), z, matrix ) , len(fg)-1)
                book['y'] = y
    
    else:
        #constructs y_p
        y = downsampling( convolution( tilda( fg.get('g') ), z, matrix ) , len(fg)-1)
        book['y'] = y

    return book

def synthesis(components, fg=None, kind='haar', matrix=True):

    """
    multi stage synthesis wavelets components  x_1, x_2, ..., x_p, y_p to 
    reproduce z

    Attributes
    ----------
    components : dict
        key values to include continuous integers from 1 to p and 'y'. Integer 
        key must correspond to x_key componenet and 'y' must correspond to 'y_p'.
        Len must be atleast 2 corresopnding to keys 1 and 'y'.
    fg : dict
        user can provide non iterative filter bank to consruct multiresolution
        analaysis of z. Default is None. If both fg and kind are provided, fg
        takes precedence. refer to function fg_construct() for structure of fg
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default
    matrix : boolean
        to use circulant matrix for calculating convolution. default is True

    Returns
    -------
    ndarray of rank at most 2
        constructed z. rank depends on rank of components of z

    Raises
    ------
    TypeError
        if len of componets is less than 2
    
    """

    if len(components)<2:
        raise ValueError("Can not restore vector from only compressed vector")

    z=0
    d = components[1]

    if d.ndim==1:
        N = len(components[1])*2
        p = find_p( N // (len(components['y'] )))
    elif d.ndim==2:
        N = d.shape[1] * 2
        p = find_p( N// components['y'].shape[1])
    #figures out N and p from components

    if fg ==None:
        fg = construct_fg(N,p,kind,matrix)

    l=1
    #reconstruction
    for x,f in zip(components.values(), fg.values()):
        temp = convolution( f , upsampling( x, l ), matrix )
        z += temp
        if l != p:
            l += 1
            
    return z


def compression( data , P=1 , kind='haar', matrix=True):
    """
    compresses an ndarray to level p.
    P!=p

    Attributes
    ----------
    data : ndarray of rank 2 or 3  
        If ndarray is of rank 2. Compression is performed along axis 0 first and
        then along axis 1 of the compressed ndarray if ndarray is of dimension 
        3, ndarray of rank 2 are creted over axis 3. Each 2darray is compressed 
        and then concatenated along the third axis
    P : int
        level of the compression. Takes poitive integer or -1 as input. If P=-1,
        p is assumed to be minimum of the highest power of 2 that divides the 
        size along axis 0 or along axis 1. If P > minimum of the highest power 
        of 2 that divides the size along axis 0 or along axis 1, P is assumed to
        be -1. Default is -1
    kind : str
        defines the type of wavelet to be constructed. Takes 3 possible values
        'haar','D6' and 'complex'. 'haar' is default
    matrix : boolean
        to use circulant matrix for calculating convolution. default is True

    Returns
    -------
    ndarray of rank at 2 or 3
        compressed data

    Raises
    ------
    TypeError
        -if rank of shape is not 2 or 3
        -if dimension of input array is divisble by 2 at axis 0 and 1 
    
    """
    

    shp = data.shape        

    if len(shp) < 2 or len(shp) > 3  :
        raise TypeError(" input array must have rank of 2 or 3 ")

    elif shp[0] % 2 != 0 or shp[1] % 2 != 0:
        raise TypeError("dimension of input array must be divisble by 2 at axis 0 and 1")

    elif len(shp) == 2:
        data = data[:,:,None]
    
        

    shp = data.shape

    row_len = shp[1]    #length of row
    col_len = shp[0]    #length of col
        
    p = min( find_p(row_len),find_p(col_len), P) if P!=-1 else min(find_p(row_len),find_p(col_len))

    arr=[] #compression along depth
    for i in range(shp[2]):

        mat = data[:,:,i]
        temp = analysis(mat, p, None, kind, matrix,True ) #compressed along rows
        mat = np.transpose(temp['y'])
        temp = analysis(mat, p, None, kind, matrix,True ) #compressed along columns
        arr.append( np.transpose(temp['y']) )             #add the compressed matrix to arr
            
    return np.dstack(arr)                                 #returns concated ndarray along the third axis

