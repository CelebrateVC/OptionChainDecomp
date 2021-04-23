# OptionChainDecomp
This tool Takes one option chain and decomposes it down to component distributions

## Call
![Call](https://i.imgur.com/ePakxWA.png)
## Put 
![Put](https://i.imgur.com/Wp1NBjq.png)
 
to find an approximation of `Pr(x)` for the probability that the underlying asset reaches price x, Such that it satisfies 

![CallEquation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextup%7BCallPrice%7D_k%3D%5Cint%20%5Ctextup%7Bmin%7D%28x-k%2C0%29*%5Ctextup%7BPr%7D%28x%29%5Cdelta%20x)

 and 
 
 ![PutEquation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextup%7BPutPrice%7D_k%3D%5Cint%20%5Ctextup%7Bmin%7D%28k-x%2C0%29*%5Ctextup%7BPr%7D%28x%29%5Cdelta%20x)
 
 `Pr(x)` is determined using recursive minimization until a sufficient fit is produced following the logic below
  
 ![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctextup%7BPr%7D%28x%29%5Csim%20%5Csum_i%28w_i*%5Caleph%28%5Cmu_i%2C%5Csigma_i%29%29%5Ctextup%7B%20s.t.%20%7D%5Csum_i%28w_i%29%3D1)
 
 The nature of the minimization is different between the two python files:
 #### parser.py
 this file will minimize changing every and all means, STD, and weights of all distributions, this is slower and generally does not lead to better results than the other file
 #### optim.py
 this file will minimize all weights and only the newest distribution's mean and STD. That is to say on iteration 1 it performs exactly the same as the other file, however after that the distriubtuion found in iteration 1 will be frozen and only it's weighting in the final output will change and so on for the next until iteration stops.



<hr>

*Regardless of the minimization used, the error function is as follows*

![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B50%7D%20%5Cbg_white%20%5Chuge%20%5Csqrt%7B%5Csum_k%5Cleft%28%5Cleft%28mark%5C_price_k-%5Csum_i%5Cleft%28%5Cint%5Cleft%5B%5Cmin%7B%28x-strike_k%2C0%29%7D*%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%5Csigma_i%5E2%7D%7D*e%5E%7B%5Cfrac%7B%28%5Cmu_i-x%29%5E2%7D%7B2%5Csigma_i%5E2%7D%7D%5Cright%5D%5Cdelta%20x*weight_i%20%5Cright%20%29%5Cright%29%5E2*%5Cln%28open%5C_interest_k%29%20%5Cright%29%5Cdiv%5Csum_k%28%5Cln%28open%5C_interest_k%29%29%7D) 
 
 That is to say:
 
 * The sum for each option k
    * the Sum for each weighted distribution i
       * the calculated price of a call as the integral above, using a Gaussian Standard Normal with given Mean and Standard Deviation
       * weighted by weighting factor 
    * that result is the estimated call price *(or put price if the specific option is a put, not pictured for brevity sake)*
    * less the mark price of the option
    * the quantity squared
    * result multiplied by the natural log of the open interest for the option
 * divided by the sum of the natural logs of the open interest
* the whole quantity under a square root 
 
 
 # Inputs
 Code takes in a MessagePack serialized format for increased space efficency from JSON the format of the file is a such
 
 ```python
{
"GME":{  # Underlying ticker 
    150: { # Strike Price of Option
      "P": RobinhoodScrape
      "C": RobinhoodScrape
    } 
  }
}
```

Where the RobinhoodScrape objects are minimally mark_price and open_interest, but for mine are 
```json
{
    "occ_symbol": "GME   210430P00187500",
    "adjusted_mark_price": 35.68,
    "ask_price": 36.65,
    "ask_size": 41,
    "bid_price": 34.7,
    "bid_size": 67,
    "break_even_price": 151.82,
    "high_price": NaN,
    "instrument": "https://api.robinhood.com/options/instruments/dd6aa0c2-e238-48fc-9f4b-dc5ab25e20fb/",
    "instrument_id": "dd6aa0c2-e238-48fc-9f4b-dc5ab25e20fb",
    "last_trade_price": 34.05,
    "last_trade_size": 1.0,
    "low_price": NaN,
    "mark_price": 35.675,
    "open_interest": 45,
    "previous_close_date": "2021-04-20",
    "previous_close_price": 37.73,
    "volume": 0,
    "symbol": "GME",
    "chance_of_profit_long": 0.484926,
    "chance_of_profit_short": 0.515074,
    "delta": "-0.711753",
    "gamma": 0.008651,
    "implied_volatility": 1.588772,
    "rho": "-0.036565",
    "theta": "-0.747129",
    "vega": 0.084762,
    "high_fill_rate_buy_price": 36.34,
    "high_fill_rate_sell_price": 34.94,
    "low_fill_rate_buy_price": 35.6,
    "low_fill_rate_sell_price": 35.68,
    "Type": "P"
}
```


Make your own fancy scraper if you want 

-or-
 
just have the devtools open (`F12` on most browsers) and filter down on GET queries for "/marketdata/options/" view the pages that you want and right click save as HAR

This HAR will be a JSON file and in \["log"\]\["entries"\]\[i\]\["responce"\]\["content"\]\["text"\] there should be a json object as text, parse it and within that \["results"\] should be a list of these objects that I have been using

all the heirarchial information is in the `occ_symbol`

* the first 6 characters contain the ticker
* the next 6 are the date in YYMMDD
* the next 1 is if it is a Put or call
* and the remaining characters are the strike price *1000
