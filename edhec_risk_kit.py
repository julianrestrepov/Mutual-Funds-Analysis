import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently as an excercise
    to the reader.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently as an excercise
    to the reader.
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """

    rf_per_period= (1 + risk_free_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index,
    the previuos peaks and
    the percentage drawdown. 
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previous Peak": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the top and bottom deciles by marketcap.
    """
    
    me_m = pd.read_csv("data/portfolios_formed_on_me_monthly_ew.csv",
                       header=0, index_col = 0, na_values=-99.99)
    
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'largeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns.
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                       header=0, index_col = 0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns_extra():
    """
    Load and format the Ken French 30 Insdustry Portfolios Value Weighted Monthly
    """
    files = ['size', 'nfirms']
    df_list = []
    for file in files:
        df = pd.read_csv(f"data/ind30_m_{file}.csv", header=0, index_col = 0)
        df.index = pd.to_datetime(df.index, format="%Y%m").to_period('M')
        df.columns = df.columns.str.strip()
        df_list.append(df)
        
    return df_list[0], df_list[1]

def get_ind_returns():
    """
    Load and format the Ken French 30 Insdustry Portfolios Value Weighted Monthly
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",
                       header=0, index_col = 0, parse_dates=True)
    ind = ind/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r= r - r.mean()
    # Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r= r - r.mean()
    # Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the jarque-Bera test to determina if a series is normal or not
    Test is appled at the 1% level by default
    Returns True if the hypothesis of normality is accepted, false otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level




def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that parameter "level" is percent of the returns
    fall below that number, an the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
    
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the z score assuming it was Gaussian
    
    z = norm.ppf(level/100)
    if modified:
        # Modify Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
             (z**2 - 1)*s/6 + 
             (z**3 -3*z)*(k-3)/24 - 
             (2*z**3 - 5*z)*(s**2)/36
            )
              
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that parameter "level" is percent of the returns
    fall below that number, an the (100-level) percent are above
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
        
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Weight -> Vol
    """

    return (weights @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
    "Returns":rets,
    "Volatility":vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)
    
    
def minimize_vol(target_return, er, cov):
    """
    target_return -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args':(er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1 
    }
    results = minimize(portfolio_vol, init_guess, args=(cov,), method="SLSQP",
                      options={'disp':False},constraints=(return_is_target, weights_sum_to_1),
                      bounds = bounds)
    
    return results.x



def optimal_weights(n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate, the expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n

    weights_sum_to_1 = {
        'type':'eq',
        'fun':lambda weights: np.sum(weights) - 1 
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
        
        
    results = minimize(neg_sharpe_ratio, init_guess, args=(riskfree_rate, er, cov), method="SLSQP",
                      options={'disp':False},constraints=(weights_sum_to_1),
                      bounds = bounds)
    
    return results.x

def gmv(cov):
    """
    Return the weight of the global Minimum vol portfolio
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
    
    

def plot_ef(n_points, er, cov, style=".-", show_cml=True, show_ew=True, riskfree_rate=0.03, show_gmv=True, ):
    """
    Plots the N-asset efficient frontier
    """
    
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    
    ef = pd.DataFrame({
    "Returns":rets,
    "Volatility":vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW
        print("EW Weights: ", w_ew)
        print("EW Return: ", r_ew)
        print("EW Vol: ", vol_ew)
        print("EW Sharpe Ratio: ", (r_ew - riskfree_rate)/vol_ew)
        print()
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o")
        
       
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display EW
        print("GMV Weights: ", w_gmv)
        print("GMV Return: ", r_gmv)
        print("GMV Vol: ", vol_gmv)
        print("GMV Sharpe Ratio: ", (r_gmv - riskfree_rate)/vol_gmv)
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o")
        
    
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker="o", linestyle="dashed");
        
    return ax



def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t, given
    interest rate r.
    """
    return 1/(1 + r)**t



def pv(l, r):
    """
    Computes the present vaue of a sequence of liabilities
    L is indexed by time, and values are the amounts of each liability.
    return the present value of the sequence
    """
    dates = l.index
    discounts = discount(dates, r)
    return (discounts*l).sum()
    
def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return assets/pv(liabilities, r)