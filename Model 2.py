from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import timedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

class DistanceMethodPairsTrading(QCAlgorithm):
    """
    Distance pairs trading method strategy with cointegration testing.

    ETF Pair Formation:
    - Using a 12 month period (formation window), we build normalised prices
      for a single pair of ETFs, compute a spread of the prices for each ETF, and calculate
      the standard deviation and mean of the spread.
    - ETFs which have at least 1 zero-volume trade day are removed, in this case, resultant
      in the backtest not continuing (as there are only two ETFs we test at a time)!
    - Additionally, we test for cointegration using the Johansen cointegration test on all
      historical data from start_date to current time, and validate stationarity of the
      cointegration residual using the Augmented Dickey-Fuller (ADF) test at 10% significance level.
    - Only pairs that pass the cointegration test proceed to the trading window.

    ETF Trading Strategy:
    - We trade over a period of 6 months.
    - We have a daily signal that we use to inform trading. We build this signal to be a z_score
      which checks how many standard deviations we are from the mean. To align with the methodology
      in the lecture, we have chosen a z-score of 2 for our initial test.
      - Thus, we enter a posiiton if z_score >= 2 into a trade, longing the cheaper stock
        and shorting the more expensive stock.
      - We exit a position when the normalised prices cross (spread flips in sign) or if it goes
        6 months without crossing.
    """

    # Parameters for our trading strategy.

    # Formation window for the ETF pairs in days.
    # Can be either 12 months of trading days or 12 months of total days.
    FORM_DAYS   = 252

    # Trading window for the ETF pairs.
    # Can be either 6 months of trading days or 6 months of total days.
    TRADE_DAYS  = 126

    # Z_Score for entering a position. Must clear this threshold, such that spread
    # must be >=2 standard deviations from the formation-period standard deviation.
    ENTRY_Z     = 2.0
    GROSS_CAP   = 0.40

    # ETF A is treated as Y, B as X.
    ETF_A = "VOO"
    ETF_B = "IVV"

    # Resolution is the duration of time used for sampling the data source.
    # Market data is sampled in 5 different resolutions: TICK, SECOND, MINUTE, HOUR, DAILY
    # For the purpose of our strategy, we want to sample daily data (of ETF prices) to do
    # daily trades.
    TRADE_FREQUENCY   = Resolution.DAILY

    # Starting date for our back-testing strategy.
    START_YEAR = 2006
    START_MONTH = 1
    START_DAY = 1

    # Starting cash for our back-testing strategy.
    START_CASH = 10000

    def Initialize(self):
        """
        This method initialises the pairs trading algorithm by:
        - Configuring the backtesting settings (dates, cash)
        - Adding the ETFs to the algorithm at the chosen trading frequency (resolution)
        - Initialising state variables for formation and trading.
        - Schedules the frequency of the trading logic to run at the specific resolution.
        """

        #1) Pick a start date for our strategy to back-test from and set our initial starting cash.
        self.set_start_date(self.START_YEAR, self.START_MONTH, self.START_DAY)
        self.set_cash(self.START_CASH)

        #2) Add both of the ETFs to the algorithm. ETF_A is mapped to 'Y', and ETF_B is mapped to 'X'.
        # This line ensures that Open, High, Low, Close, Volume data at the given trade frequency (daily in our case)
        # is obtained and added to our algorithm such that we can begin trading and holding positions.
        self.y = self.add_equity(self.ETF_A,  self.TRADE_FREQUENCY).symbol
        self.x = self.add_equity(self.ETF_B, self.TRADE_FREQUENCY).symbol

        #3) Initialize the state tracking variables. These ensure that we are either only in formation mode or trading mode, but not both
        # at the same time.
        self.in_trade_window   = False
        self.trade_window_end  = None
        self.form_window_start = None

        #4) The variables for our algorithm are initialised as none for each ETF.
        self.y0 = None
        self.x0 = None

        #5) The formation statistics for the mean and standard deviation are initialised as none.
        self.form_mean = None
        self.form_std  = None

        #6) The tracking variable which tracks if prices cross is initialised to none.
        # This variable uses the previous spread of prices to compare against the current spread for
        # price crossing.
        self.prev_spread = None

        #7) Initialize historical data storage for cointegration testing.
        # This will store the complete price history from start_date to current time,
        # which is required for the Johansen cointegration test and ADF stationarity test.
        self.historical_data = None

        #8) This ensures that QuantConnect calls our Trading Logic at a specific time, in this case every TRADING day
        # (not calendar day) when y (ETF_A) is trading, 1 minute after the market opens.
        self.schedule.on(
            self.date_rules.every_day(self.y),
            self.time_rules.after_market_open(self.y, 1),
            self.RunTradingLogic
        )

        #9) Print a debug to show that we have initialised our ETF pairs successfuly with cointegration testing.
        self.debug("Initialised ETF pairs with cointegration testing: {} and {}".format(self.ETF_A, self.ETF_B))

    def RunTradingLogic(self):
        """
        This method runs the trading logic for the pairs strategy by:
        - Validating that both ETFs have data for the day
        - Switching between formation and trading modes
        - Starting and ending trade windows, with liquidations when required
        - Computing normalised prices, spreads, and z-scores
        - Detecting spread crossings to trigger exits
        - Checking entry conditions and opening long/short positions with balanced weights
        - Updating the previous spread each day for future crossing detection
        """

        #1) Run a data validation check to ensure both securities have valid data.
        # self.securities is a dictionary of all the ETFs initialised using add_equity in the Initialize
        # method. has_data ensures that there are valid data points for each ETF given the resolution of trading.
        if not (self.securities[self.y].has_data and self.securities[self.x].has_data):
            return

        """
        2) Checks whether we are currently not in a trading window. If true, we are in formation mode
        and should recompute the formation stats.
        """
        if not self.in_trade_window:

            #2a) Calls the formation method to build the sample of ETFs, normalise prices, compute spread mean and standard deviation,
            # check for liquidity issues, test for cointegration, and returns true if successful or otherwise false.
            formation = self.BuildFormationWindow()

            """
            2b) Upon successful formation:
                1. We flip the mode to trading from formation mode.
                2. We set a stop date for the trading window equal to our current time plus the length of the period we would
                    like the trade window to be (in days).
                3. We reset the previous spread of prices to none, as we are now at the beginning of a new trading period.
                4. We print a debug to be aware of the start of a new trading period, highlighting the start data, formation mean and standard
                    deviation of the price spread, and the normalisation prices of each ETF.
            """
            if formation:
                self.in_trade_window  = True
                # timedelta turns the integer into calendar days and adds it to the current time to obtain the trade_window_end.
                self.trade_window_end = self.time + timedelta(days=int(self.TRADE_DAYS * 1.5))
                self.prev_spread = None
                self.debug(f"New Trade Window  | Time: {self.time.date()} | Mean: {self.form_mean:.4f} Standard Deviation: {self.form_std:.4f} Y/ETF_A Normalised Price:{self.y0:.2f} X/ETF_B Normalised Price:{self.x0:.2f}")
            return

        #3) Check if a trade window end date was set and that the current time is at or past that date.
        # If the date has elapsed, then we want to exit all trades.
        if self.trade_window_end and self.time >= self.trade_window_end:

            #3a) Check our portfolio to see if we have anything invested in either of the two ETFs.
            # and liquidate the position, adding a tag explaining why it was liquidated.

            # self.portfolio keeps a record of every security owned and related information such as quantity, average price, total cost, unrealised gains.
            # self.invested returns True if a position (non-zero) is held in a specific security, and false if there is a non-zero position held.
            # self.liquidate closes all open positions by entering opposing trades. For example, if currently long it will place asell order, if short it will place a buy order.
            if self.portfolio[self.y].invested or self.portfolio[self.x].invested:
                self.liquidate(self.y, "Trade window has ended.")
                self.liquidate(self.x, "Trade window has ended.")

            #3b) Resets our in_trade_window tracking variable to be false once we have left the trade window, and prints
            # a debug to the console giving the date of the end of the trade window with a message.
            self.in_trade_window = False
            self.debug(f"Date: {self.time.date()} | Trade window has ended.")
            return

        #4) Reads the most recent close prices (in decimal form) and updates the ETF variables
        # to reflect these close prices. If our trading algorithm runs on open, it would thus use yesterday's close prices!
        y_price = float(self.securities[self.y].close)
        x_price = float(self.securities[self.x].close)

        #5) If our algorithim has not been initialised for whatever reason, then we have no
        # formation prices to obtain normalised prices. Early return in this case, and if the prices
        # are invalid (negative).
        if self.y0 is None or self.x0 is None or self.y0 <= 0 or self.x0 <= 0:
            return

        #6) Obtain the current day's normalised prices for each ETF, along with the normalised price spread.
        y_norm = y_price / self.y0
        x_norm = x_price / self.x0
        spread = y_norm - x_norm

        #7) If the formation standard deviation has not been initialised, or is 0 (invalid), early return.
        if self.form_std is None or self.form_std == 0:
            return

        #8) Obtain the current day's z-score.
        z = (spread - self.form_mean) / self.form_std

        #9) Check if we already hold a position in EITHER of the ETF Pairs.
        invested = self.portfolio[self.y].invested or self.portfolio[self.x].invested

        #10) If we currently hold a position in either ETF and we already have a previous spread 
        # recorded (i.e. not the very first day of the trading window), then check for a price 
        # crossing. A crossing is detected if the sign of the spread changes.
        # 
        # The product of the two spreads, (prev_spread * spread) will be negative (< 0) in the event of
        # a sign change. Thus, this is our condition to check for a crossing of prices.

        # In the case that all 3 conditions are met, we liquidate both positions and update prev_spread
        # to the current spread. Early return after closing the positions given a crossing of prices.
        if invested and self.prev_spread is not None and (self.prev_spread * spread < 0):
            self.liquidate(self.y, "Prices have crossed.")
            self.liquidate(self.x, "Prices have crossed.")
            self.debug(f"Date: {self.time} | Closing position given prices crossing | Z-Score: {z:.2f} | Spread: {spread:.4f}")
            self.prev_spread = spread
            return

        #11) If we don't hold a position in either ETF, and the magnitude of the current Z-Score >= Entry Z-Score,
        # we consider opening a new trade. abs(z) ensures that the sign of the Z-Score is irrelevant here, as we
        # only care about magnitude.
        if not invested and abs(z) >= self.ENTRY_Z:

            #11a) Obtain the current total portfolio value.
            pv = self.portfolio.total_portfolio_value

            #11b) Compute the per-leg weight: each ETF (long and short) gets half of the total gross cap, which we initialise
            # in our parameters.
            # Example: with GROSS_CAP = 0.40, each ETF leg is 0.20 (20% long, 20% short).
            # The min(..., 0.49) is a safeguard so no single ETF leg ever exceeds 49% of portfolio value,
            # even if GROSS_CAP is set too high. This ensures balanced exposure and prevents over-allocation.
            per_leg_weight = min(self.GROSS_CAP / 2.0, 0.49)

            #11c) The type of position we would like to enter in is based on which normalised price is higher.
            # A positive spread (> 0) spread means that Y_Norm (ETF_A) > X_Norm (ETF_B), thus Y > X (relatively).
            # Hence, we go long on X (ETF_B, short on Y (ETF_A). Opposite logic occurs if the spread is negative (< 0).
            # We enter into the trades with 'per_leg_weight' of the portfolio value per ETF. set_holdings calculates
            # the number of asset units to purchase based on the aforementioned weighting!
            if spread > 0:
                self.set_holdings(self.x, +per_leg_weight)
                self.set_holdings(self.y, -per_leg_weight)
                side = f"Longing {self.ETF_B} / Shorting {self.ETF_A}"
            else:
                self.set_holdings(self.y, +per_leg_weight)
                self.set_holdings(self.x, -per_leg_weight)
                side = f"Longing {self.ETF_A} / Shorting {self.ETF_B}"

            #11d) Print to the console our trade and update our previous spread. Return, as we have traded for the day.
            self.debug(f"Date: {self.time} | {side} | Z-Score: {z:.2f} Spread: {spread:.4f}")
            self.prev_spread = spread
            return

        #12) Always update the previous spread.
        self.prev_spread = spread

    def BuildFormationWindow(self) -> bool:
        """
        Builds the formation window statistics for the ETF pair by:
        - Pulling historical OHLCV data for both ETFs over the formation period
        - Validating data completeness and liquidity (no NaNs, no zero-volume days)
        - Normalising each ETF's price series to its first-day price
        - Calculating the spread, mean, and standard deviation of the normalised series
        - Testing for cointegration using the Johansen test on all historical data
        - Validating stationarity of the cointegration residual using ADF test at 10% significance
        - Saving the formation statistics and baseline prices for later trading
        Returns True if the formation stats are valid (non-zero std) and pair is cointegrated, otherwise False.
        """

        #1) Request historical data for Open, Close, Low, High, Volume information for both ETFs over the formation window.
        # self.history() asks QuantConnect for past data (in this case, FORM_DAYS worth at TRADE_FREQUENCY resolution).
        # The result is a  DataFrame with multi-index (time + symbol) information, containing price and volume data.
        bars = self.history([self.y, self.x], self.FORM_DAYS, self.TRADE_FREQUENCY).copy()

        #2) If the dataframe is empty, return false.
        if bars.empty:
            return False

        #3) Restructure the historical data to get a cleand up time series of the close prices and volume per ETF.
        # If the restructuring fails, return False.
        try:
            # Restructures the data frame to have only time, close prices, per ETF.
            close = bars['close'].unstack(level=0)
            vol   = bars['volume'].unstack(level=0)
        except Exception:
            return False

        #4) Returns false if either ETF doesn't exist in the historical data.
        if self.y not in close.columns or self.x not in close.columns:
            return False

        #5) Checks for zero-volume days for either ETF. If any zero-volume days exist, we cannot form data (validity check).
        # Missing values are replaced with 0, and sets 0 values to equal True. If a True exists, we skip formation and return False.
        if vol[self.y].fillna(0).eq(0).any() or vol[self.x].fillna(0).eq(0).any():
            self.debug("Formation skipped due to zero-volume day in history")
            return False

        #6) Removes any rows where one or both of the ETFs have missing values.
        # Such as, if only one ETF has data on one day and the other doesn't.
        close = close.dropna()

        #7) Counts how many rows of data exist. If the number of rows is less than the
        # formation days we expect in a period, then the formation history is incomplete.
        # Returns false when incomplete.
        if len(close) < self.FORM_DAYS:
            return False

        #8) Creates a series for each ETF indexed by data, with close price values, as a float.
        y_series = close[self.y].astype(float)
        x_series = close[self.x].astype(float)

        #9) Obtains the very first day of the formation window to get the starting price for both ETFs.
        y0 = float(y_series.iloc[0])
        x0 = float(x_series.iloc[0])

        #10) If the starting price is negative, then it is incorrect and we return False.
        if y0 <= 0 or x0 <= 0:
            return False

        #11) Obtain the normalised prices and the spread of these normalised prices for the formation period.
        y_norm = y_series / y0
        x_norm = x_series / x0
        spread = y_norm - x_norm

        # ===========================================================================================
        # COINTEGRATION TESTING SECTION
        # ===========================================================================================
        # This section tests whether the two ETFs are cointegrated using statistical methods.
        # Cointegration means that even though the individual price series may be non-stationary,
        # there exists a linear combination of them that IS stationary. This is essential for
        # pairs trading because it suggests the spread will mean-revert.
        #
        # We use two tests:
        # 1. Johansen Cointegration Test - finds the cointegration vector (weights w1, w2)
        # 2. Augmented Dickey-Fuller Test - verifies the resulting residual is stationary
        # ===========================================================================================

        #12) Print debug information showing current time and start date for cointegration testing context.
        self.debug(f"Current time = {self.time}; start_date = {self.start_date}")

        #13) Request complete historical data from the backtest start_date to current time for cointegration testing.
        # Unlike the formation window (step 1), this pulls ALL available historical data, not just FORM_DAYS.
        # This longer history provides more robust cointegration testing.
        hist = self.history([self.y, self.x], self.start_date, self.time, self.TRADE_FREQUENCY).copy()
        
        # If no historical data is available, we cannot perform cointegration testing, so return False.
        if hist.empty:
            self.debug("No historical data available for cointegration testing.")
            return False

        #14) Extract and clean the historical close prices for cointegration testing.
        # This process ensures both ETFs have data on the same dates (aligned time series).
        try:
            # Unstack converts the multi-index DataFrame to have symbols as columns.
            close_hist = hist['close'].unstack(level=0)
            
            # Keep only the two ETFs we're testing and drop any rows where either ETF has missing data.
            # This ensures our cointegration test only uses dates where BOTH ETFs have valid prices.
            close_hist = close_hist[[self.y, self.x]].dropna()
        except Exception as e:
            self.debug(f"Error processing historical closes for cointegration: {e}")
            return False

        #15) Store the cleaned historical data as a DataFrame for the Johansen test.
        # self.historical_data will contain aligned close prices for both ETFs across all available history.
        self.historical_data = close_hist.copy()
        closes_pair = close_hist[[self.y, self.x]]

        #16) Prepare the data for the Johansen cointegration test.
        # We create a 2D numpy array where each column represents one ETF's price series.
        # Note: We use raw prices here, NOT normalized prices, as the Johansen test
        # is designed to work with the actual price levels.
        raw_prices = np.column_stack((closes_pair[self.y].values, closes_pair[self.x].values))

        #16a) Verify that both price series are I(1) before running Johansen test.
        # I(1) means: non-stationary in levels, but stationary in first differences.
        # This is a critical assumption for the Johansen cointegration test.
        
        # Test if price levels are non-stationary (should NOT reject null hypothesis of unit root)
        adf_y_levels = adfuller(raw_prices[:, 0])
        adf_x_levels = adfuller(raw_prices[:, 1])
        
        # Test if first differences are stationary (should reject null hypothesis of unit root)
        y_diff = np.diff(raw_prices[:, 0])
        x_diff = np.diff(raw_prices[:, 1])
        adf_y_diff = adfuller(y_diff)
        adf_x_diff = adfuller(x_diff)
        
        # Check I(1) conditions at 5% significance level:
        # 1. Levels should be non-stationary: |test_stat| < |critical_value_5%|
        # 2. First differences should be stationary: |test_stat| >= |critical_value_5%|
        y_levels_nonstationary = abs(adf_y_levels[0]) < abs(adf_y_levels[4]['5%'])
        x_levels_nonstationary = abs(adf_x_levels[0]) < abs(adf_x_levels[4]['5%'])
        y_diff_stationary = abs(adf_y_diff[0]) >= abs(adf_y_diff[4]['5%'])
        x_diff_stationary = abs(adf_x_diff[0]) >= abs(adf_x_diff[4]['5%'])
        
        # Both series must be I(1) for Johansen test to be valid
        if not (y_levels_nonstationary and x_levels_nonstationary and y_diff_stationary and x_diff_stationary):
            self.debug(f"I(1) assumption violated:")
            self.debug(f"  {self.ETF_A} levels non-stationary: {y_levels_nonstationary} (tau={adf_y_levels[0]:.4f})")
            self.debug(f"  {self.ETF_B} levels non-stationary: {x_levels_nonstationary} (tau={adf_x_levels[0]:.4f})")
            self.debug(f"  {self.ETF_A} diff stationary: {y_diff_stationary} (tau={adf_y_diff[0]:.4f})")
            self.debug(f"  {self.ETF_B} diff stationary: {x_diff_stationary} (tau={adf_x_diff[0]:.4f})")
            return False

        #17) Perform the Johansen cointegration test.
        # 
        # IMPORTANT ASSUMPTION: The Johansen test assumes both time series are I(1), meaning:
        #   - Non-stationary in levels (the raw prices have unit roots)
        #   - Stationary in first differences (price changes are stationary)
        # This is typically true for financial asset prices, which follow random walks.
        # The test then checks if there exists a linear combination that is I(0) (stationary).
        #
        # Parameters:
        #   - self.historical_data: the DataFrame containing both ETF price series
        #   - det_order=0: assumes no deterministic trend in the cointegration relationship
        #   - k_ar_diff=1: uses 1 lag in the vector autoregression
        # 
        # The test returns a result object containing eigenvalues, eigenvectors (cointegration vectors),
        # and test statistics. The first eigenvector (evec[:, 0]) represents the cointegration weights.
        coint_result = coint_johansen(self.historical_data, det_order=0, k_ar_diff=1)
        
        # Extract the cointegration vector components (weights for each ETF).
        # w1 is the weight for ETF_A (y), w2 is the weight for ETF_B (x).
        w1 = coint_result.evec[0, 0]
        w2 = coint_result.evec[1, 0]

        #18) Compute the cointegration residual (epsilon) using the weights from the Johansen test.
        # epsilon = w1 * price_series_A + w2 * price_series_B
        # 
        # This residual represents the "spread" after accounting for the cointegration relationship.
        # If the ETFs are truly cointegrated, this residual should be stationary (mean-reverting).
        epsilon = w1 * raw_prices[:, 0] + w2 * raw_prices[:, 1]

        #19) Perform the Augmented Dickey-Fuller (ADF) test on the cointegration residual.
        # The ADF test checks whether a time series is stationary.
        # 
        # Returns: (test_statistic, p_value, lags_used, nobs, critical_values, icbest)
        # We're primarily interested in:
        #   - adf_result[0]: the test statistic (more negative = more likely stationary)
        #   - adf_result[4]: dictionary of critical values at different significance levels
        adf_result = adfuller(epsilon)

        #20) Check if the residual passes the stationarity test at 10% significance level.
        # For the ADF test, a MORE NEGATIVE test statistic indicates stronger evidence of stationarity.
        # We compare the absolute value of our test statistic against the 10% critical value.
        #
        # If |test_statistic| >= |critical_value_10%|, the residual is stationary (good for pairs trading).
        # If not, the pair is not cointegrated enough to trade, so we reject this formation and return False.
        if not abs(adf_result[0]) >= abs(adf_result[4]['10%']):
            self.debug(f"Pair not statistically significant: tau = {abs(adf_result[0]):.4f}, 10% S.L = {abs(adf_result[4]['10%']):.4f}")
            self.debug(f"Total data points per ticker = {len(self.historical_data)}")
            return False

        #21) Saves the mean, standard deviation of the normalised prices, along with the starting prices
        # of each ETF. Additionally, saves the timestamp of the first day of the formation window.
        self.form_mean = float(spread.mean())
        self.form_std  = float(spread.std(ddof=1))
        self.y0 = y0
        self.x0 = x0
        self.form_window_start = close.index[0]

        #22) Returns true if our standard deviation is non-zero. We only want to trade if this is the case. Otherwise, returns false.
        return self.form_std > 0