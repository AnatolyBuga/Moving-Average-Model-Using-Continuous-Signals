import math
import polars as pl

def yearfrac(start, end):
    '''Assumes Act 360'''
    return (start-end).days / 360

class EquallyWeightedPortfolio:
    """
        polars contexts such as with_columns are executed in parallel, making it extremely efficient
        https://stackoverflow.com/questions/71105136/expression-in-polars-select-context-that-refers-to-earlier-alias
    """
    storage: pl.DataFrame

    def __init__(self, st_vol: tuple[int, float], lt_vol: tuple[int, float], 
                 first_ewma: int, n_ewma: int = 4, vol_adj: int = 16,
                 rolling_avg_window=256, signal_mult=10, capsignal = (20,-20),
                 mov_avg_capsignal_weight=None,
                 posi_numer_mult = 1,
                 posi_denom_mult = 10
                 ) -> None:
        
        self.st_vol = st_vol
        self.lt_vol = lt_vol
        self.ewma_params = self._derive_ewma_params(first_ewma, n_ewma)
        self.vol_adj = vol_adj
        self.rolling_avg_window = rolling_avg_window
        self.signal_mult = signal_mult
        self.capsignal = capsignal
        self.mov_avg_capsignal_weight = mov_avg_capsignal_weight if mov_avg_capsignal_weight else self._derive_mov_avg_capsignal_weight(n_ewma)
        self.posi_numer_mult = posi_numer_mult
        self.posi_denom_mult = posi_denom_mult
    
    def _derive_mov_avg_capsignal_weight(self, n_ewma):
        n_of_signals = n_ewma - 1
        weights = [1/n_of_signals]*n_of_signals
        print("derived weights for avg Signal capfloor: " + str(weights))
        return weights

    def _derive_ewma_params(self, ewma, n_ewma):
        ewma_params = []
        ewma_val = ewma
        for _ in range(n_ewma):
            ewma_params.append((ewma_val, 2/(ewma_val+1)))
            ewma_val = ewma_val*4
        
        print("derived ewma parameters: " + str(ewma_params))
        return ewma_params
    
    def run(self, data: pl.DataFrame, store_results=True, assets: "list[str]|None" = None,
            capital: float = 100000, target_vol: float = 0.2, multiplier: float = 1):

        assets = tuple(filter(lambda col_name: col_name != 'index', data.columns )) if not assets else assets

        data = data.select([pl.col('index')] + [pl.col(name).cast(pl.Float64) for name in assets])

        res = ( 
                data.sort(by='index', descending=True)
                    # Returns
                    .with_columns(self._returns(assets)) 
                    # Moving vols
                    .with_columns(self._moving_vols(assets))
                    # Smooth Vol
                    .with_columns(self._smooth_vol(assets)) 
                    # EWMAs
                    .with_columns(self._ewmas(assets)) 
                    .with_columns(self._unpack_ewmas(assets))
                    # Diff 
                    .with_columns(self._diffs(assets))
                    #VolAdjD
                    .with_columns(self._vol_adj_d(assets))
                    #AbsVAD
                    .with_columns(self._abs_vol_adj_d(assets))
                    #AvgVAD
                    .with_columns(self._avg_vad(assets))
                    #Signal
                    .with_columns(self._signal(assets))
                    #CapSignal
                    .with_columns(self._cap_signal(assets))
                    # Avg Cap Signal
                    .with_columns(self._avg_cap_signal(assets))
                    # Posi Signals
                    .with_columns(self._posi_signals(assets, capital, target_vol, multiplier))
                    # Abs Weighted Posi Signals and DailyPnLPosiCapSignal
                    .with_columns(self._abs_weighted_posi_signals(assets)
                                  +self._daily_pnl_posi_cap_signals(assets))
                    # Drop infs
                    .with_columns(self._drop_inf_for_daily_pnl_posi_cap_signals(assets))
                    # CumSum Daily PnL
                    .with_columns(self._cumsum_daily_pnl(assets)
                    # Rolling Std PnL Cap Signal
                                    + self._pnl_cap_sig_vol(assets))
                    # Multi Cap Signal (TargVol divided by Std PnL Cap Signal)
                    .with_columns(self._multi_div_posi_return(assets, target_vol))
                    # #  Multi * Daily PnL
                    .with_columns(self._multi_times_daily_pnl_posi_cap_signals(assets))
                    # # clean nans/infs/nulls
                    .with_columns(self._drop_inf_for_multi(assets))
                    # # CumSum (Multi*DPnL)
                    .with_columns(self._cumsum_multi(assets)
                    # Cum Prod Multi        
                                  +self._cumprod_multi(assets))
                )
        
        if store_results:
            self.storage = (assets, res)

        return res

    def compute_percentages(self) -> dict:
        """Can only be executed after run'
            NOTE: Uses different YEARFRAC convention - Act/360
        """

        res = dict()

        for name in self.storage[0]:
            idx = ['index']
            name_cumprods = []
            for d in range(1, len(self.ewma_params)):
                name_cumprods.append( pl.col(name+'_CumProdMulti_'+str(d)) )
            name_cumprods.append( pl.col(name+'_CumProdMulti') )
            df = self.storage[1].filter(pl.col(name).is_not_null()).select(idx+name_cumprods)
            top = df.head(1).select(name_cumprods)
            yf = yearfrac(df['index'][0], df['index'][-1])

            percentages = top.select((pl.col('*')/100)**(1/yf) - 1)
            res[name] = percentages[0].rows()[0]

        return res


    def _returns(self, assets) -> pl.Series:
        return [(pl.col(name)/pl.col(name).shift(-1)).log().alias(name + "_Return") for name in assets]
    
    def _rolling_std(self, s: pl.Series, offset: int) -> pl.Series:
        """Polars in-built returns nans when not enough elements in the window. Python UDF is slow but allows us to keep those elements

            TODO: performance can be significantly improved with Rust, since Python UDF (User Defined Function) is GIL bounded

        Args:
            s (pl.Series): Returns

        Returns:
            pl.Series: Vol for each element's window
        """
        res = []
        max_idx = len(s) - 1

        vol_multiplier = math.sqrt(256)
        for i, v in enumerate(s):
            window_end = i + offset if i+offset<=max_idx else max_idx
            std = s[i:window_end].std(ddof=0) #compute STD.P of the window
            if std is None:
                res.append(float('nan'))
            else:
                res.append(std*vol_multiplier)
            
        return pl.Series(res)
            
    def _rolling_std_st(self, s: pl.Series) -> pl.Series:
        return self._rolling_std(s, self.st_vol[0])
    
    def _rolling_std_lt(self, s: pl.Series)-> pl.Series:
        return self._rolling_std(s, self.lt_vol[0])
        
    def _moving_vols(self, assets: list[str]) -> list[pl.Expr]:
        """ST and LT vols for each asset

        Args:
            assets (list[str]): names of assets

        Returns:
            list[pl.Expr]: Exprs (executed only when Lazy Frame is collected)
        """
        moving_vols = [ pl.col(name+'_Return').map_batches(f).alias(name+alias) 
                   for name in assets 
                   for f, alias in ( (self._rolling_std_st, '_ST_Vol'), (self._rolling_std_lt, '_LT_Vol') ) ]        
        return moving_vols
    
    def _smooth_vol(self, assets):
        """Smooth Vol for each asset"""
        return [(self.st_vol[1]*pl.col(name+'_ST_Vol')+self.lt_vol[1]*pl.col(name+'_LT_Vol')).alias(name + "_SmoothVol") for name in assets]
    
    def _ewmas(self, assets: list[str]) -> list[pl.Expr]:
        """
        Build EWMA series for each asset
        """
        return [pl.col(name).map_batches(lambda s: self._ewma(s)).alias(name+'_EWMA') for name in assets]
    
    def _ewma(self, price: pl.Series) -> pl.Series:
        """ Iterate only once per asset
        
        TODO Improve performance with Rust

        Args:
            price (pl.Series): asset price
            weight (float): EWMA weight

        Returns:
            pl.Series: EWMA series
        """
        res = []
        price_iter = iter(enumerate(reversed(price))) # start from end
        # Where price is None, EWMA is none
        while (x:=next(price_iter)) and x[1] is None:
            res.append(tuple(float('nan') for _, weight in self.ewma_params) )

        # First occurance of price - we don't multiple by previous EWMA
        _, first_price = x
        res.append(tuple(first_price*weight for _, weight in self.ewma_params) )

        for i, p in price_iter:
            res.append( tuple(p*weight + (1-weight)*res[i-1][j] for j, (_, weight) in enumerate(self.ewma_params)) )

        return pl.Series(reversed(res))
    
    def _unpack_ewmas(self, assets: list[str]) -> list[pl.Expr]:
        return [pl.col(name+'_EWMA').list.get(j).alias(name+'_EWMA_'+str(tag)) for name in assets for j, (tag, _) in enumerate(self.ewma_params)]
    
    def _diffs(self, assets: list[str]) -> list[pl.Expr]:
        res = []
        for name in assets:
            iter = enumerate(self.ewma_params)
            next(iter) # skip 1
            for j, (tag, weight) in iter:
                prev_tag = pl.col(name+'_EWMA_'+str(self.ewma_params[j-1][0]))
                current_tag = pl.col(name+'_EWMA_'+str(tag))
                res.append((prev_tag-current_tag).alias(name+'_DIFF_'+str(j)))
        return res
            
    def _vol_adj_d(self, assets: list[str]) -> list[pl.Expr]:
        """Smooth vol can be 0. Where LT and ST Vols are both 0. 
            Then, Div by 0 gives inf and breaks VolAdjD and results in inf.
            Hence, replace inf with nan

        Args:
            assets (list[str]): _description_

        Returns:
            list[pl.Expr]: _description_
        """
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                vol_adj_with_infs = ( pl.col(name + "_DIFF_"+str(d))*self.vol_adj/pl.col(name + "_SmoothVol")/pl.col(name) )
                res.append(pl.when(vol_adj_with_infs.is_infinite()).then(None).otherwise(vol_adj_with_infs).alias(name+'_VolAdjD_'+str(d)))       
                           
        return res
    
    def _abs_vol_adj_d(self, assets: list[str]) -> list[pl.Expr]:
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append ( pl.col(name+'_VolAdjD_'+str(d)).abs().alias(name+'_AbsVAD_'+str(d)) )
        return res
    
    def _rolling_avg(self, s: pl.Series) -> pl.Series:
        """Averages of window sizes for each element

            TODO: performance can be significantly improved with Rust, since Python UDF (User Defined Function) is GIL bounded

        Args:
            s (pl.Series): Absolute VolAdjD

        Returns:
            pl.Series: AvgVAD
        """
        offset = self.rolling_avg_window
        res = []
        max_idx = len(s) - 1

        for i, v in enumerate(s):
            window_end = i + offset if i+offset<=max_idx else max_idx
            mean = s[i:window_end].mean() 
            res.append(mean)
            
        return pl.Series(res)
    
    def _avg_vad(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append ( pl.col(name+'_AbsVAD_'+str(d)).map_batches(lambda s: self._rolling_avg(s)).alias(name+'_AvgVAD_'+str(d)) )
        return res
    
    def _signal(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append ( (pl.col(name+'_VolAdjD_'+str(d))*self.signal_mult/pl.col(name+'_AvgVAD_'+str(d)) ).alias(name+'_Signal_'+str(d)) )
        return res
    
    def _cap_signal(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append ( 
                    pl.max_horizontal(
                        pl.min_horizontal(pl.col(name+'_Signal_'+str(d)), pl.lit(self.capsignal[0]) ),
                        pl.lit(self.capsignal[1])
                    )
                    .alias(name+'_CapSignal_'+str(d)) 
                    )
        return res
    
    def _avg_cap_signal(self, assets: list[str]):
        res = []
        for name in assets:
            weighted_cap_signals = []
            for d in range(1, len(self.ewma_params)):
                weighted_cap_signals.append ( 
                    pl.col(name+'_CapSignal_'+str(d))*self.mov_avg_capsignal_weight[d-1]
                    ) 
            res.append(pl.sum_horizontal(weighted_cap_signals).alias(name+'_AvgCapSignal'))
        return res
    
    def _posi_signals(self, assets: list[str], capital: float, tvol: float, multiplier: float):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append(
                    (pl.col(name+'_CapSignal_'+str(d))*capital*tvol*self.posi_numer_mult/pl.col(name)/pl.col(name + "_SmoothVol")/multiplier/self.posi_denom_mult).alias(name+'_PosiCapSignal_'+str(d))
                )
            res.append(
                    (pl.col(name+'_AvgCapSignal')*capital*tvol*self.posi_numer_mult/pl.col(name)/pl.col(name + "_SmoothVol")/multiplier/self.posi_denom_mult).alias(name+'_PosiAvgCapSignal')
                )
        return res
    
    def _abs_weighted_posi_signals(self, assets: list[str] ):
        res = []
        for name in assets:
            res.append(pl.col(name+'_PosiAvgCapSignal').abs().alias(name+'_AbsPosiAvgCapSignal'))
        return res
    
    def _daily_pnl_posi_cap_signals(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append(
                    (pl.col(name+'_PosiCapSignal_'+str(d)).shift(-1)*pl.col(name+'_Return')).alias(name+'_DailyPnLPosiCapSignal_'+str(d))
                )
            res.append(
                    (pl.col(name+'_PosiAvgCapSignal').shift(-1)*pl.col(name+'_Return')).alias(name+'_DailyPnLPosiAvgCapSignal')
                )
            
        return res
    
    def _infs_nans_nulls_to_zero(self, assets, col_name, avg_col):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                col = pl.col(name+col_name+str(d))
                res.append(
                    pl.when(col.is_infinite().or_(col.is_nan(), col.is_null())).then(None).otherwise(col).keep_name()
                )
            col = pl.col(name+avg_col)
            res.append(
                    pl.when(col.is_infinite().or_(col.is_nan(), col.is_null())).then(None).otherwise(col).keep_name()
                )
        return res

    def _drop_inf_for_daily_pnl_posi_cap_signals(self, assets: list[str]):
        return self._infs_nans_nulls_to_zero(assets, '_DailyPnLPosiCapSignal_', '_DailyPnLPosiAvgCapSignal')
    
    def _cumsum_daily_pnl(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                col = pl.col(name+'_DailyPnLPosiCapSignal_'+str(d))
                res.append(
                    col.reverse().cumsum().reverse().alias(name+'_CumSumDailyPnLPosiCapSignal_'+str(d))
                )

            col = pl.col(name+'_DailyPnLPosiAvgCapSignal')
            res.append(
                    col.reverse().cumsum().reverse().alias(name+'_CumSumDailyPnLPosiAvgCapSignal')
                )
        return res
    
    def _pnl_cap_sig_vol(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                # Reuse _rolling_std_st since window is 44
                res.append ( pl.col(name+'_DailyPnLPosiCapSignal_'+str(d)).map_batches(lambda s: self._rolling_std_st(s)).alias(name+'_StdDailyPnLPosiCapSignal_'+str(d)) )
            res.append ( pl.col(name+'_DailyPnLPosiAvgCapSignal').map_batches(lambda s: self._rolling_std_st(s)).alias(name+'_StdDailyPnLPosiAvgCapSignal') )
        return res
    
    def _multi_div_posi_return(self, assets: list[str], target_vol):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append ( (target_vol/pl.col(name+'_StdDailyPnLPosiCapSignal_'+str(d))).alias(name+'_MultiStdDailyPnLPosiCapSignal_'+str(d)) )
            res.append ( (target_vol/pl.col(name+'_StdDailyPnLPosiAvgCapSignal')).alias(name+'_MultiStdDailyPnLPosiAvgCapSignal') )
        return res
    
    def _multi_times_daily_pnl_posi_cap_signals(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                res.append(
                    (pl.col(name+'_MultiStdDailyPnLPosiCapSignal_'+str(d)).shift(-1)*pl.col(name+'_DailyPnLPosiCapSignal_'+str(d))).alias(name+'_MultiStdTimesDailyPnL_'+str(d))
                )
            res.append(
                    (pl.col(name+'_MultiStdDailyPnLPosiAvgCapSignal').shift(-1)*pl.col(name+'_DailyPnLPosiAvgCapSignal')).alias(name+'_AVGMultiStdTimesDailyPnL')
                )
            
        return res
    
    def _drop_inf_for_multi(self, assets: list[str]):
        return self._infs_nans_nulls_to_zero(assets, '_MultiStdTimesDailyPnL_', '_AVGMultiStdTimesDailyPnL')
    
    def _cumsum_multi(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                col = pl.col(name+'_MultiStdTimesDailyPnL_'+str(d))
                res.append(
                    col.reverse().cumsum().reverse().alias(name+'_CumSumMulti_'+str(d))
                )

            col = pl.col(name+'_AVGMultiStdTimesDailyPnL')
            res.append(
                    col.reverse().cumsum().reverse().alias(name+'_AVGCumSumMulti')
                )
        return res
    
    def _cumprod_multi(self, assets: list[str]):
        res = []
        for name in assets:
            for d in range(1, len(self.ewma_params)):
                col = 1+pl.col(name+'_MultiStdTimesDailyPnL_'+str(d))
                res.append(
                    (100*col.reverse().cumprod().reverse()).alias(name+'_CumProdMulti_'+str(d))
                )

            col = 1+pl.col(name+'_AVGMultiStdTimesDailyPnL')
            res.append(
                    (100*col.reverse().cumprod().reverse()).alias(name+'_CumProdMulti')
                )
        return res

