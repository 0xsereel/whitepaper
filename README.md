# Sereel Protocol Whitepaper Repository

This repository contains the official whitepaper and supporting materials for the **Sereel Protocol**.

## Overview

The Sereel Protocol is a novel DeFi infrastructure designed specifically for emerging markets, featuring:

- **Cross-module synergies** between AMM, lending, and options modules
- **Capital efficiency improvements** of up to 2.5x through innovative vault architecture
- **Risk management framework** tailored for emerging market volatility
- **Institutional-grade** security and compliance features

## Repository Contents

- `surreal.pdf` - The complete whitepaper document
- `surreal.py` - Core Python implementation and testing suite (todo: Solidity)

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed with the following packages:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Running the Test Suite

The repository includes comprehensive testing scripts that regenerate all the results presented in the whitepaper.

#### Core Protocol Tests (`surreal.py`)

```bash
python surreal.py
```

This script runs the basic protocol functionality tests including:
- Vault initialization and configuration
- Individual module performance analysis
- Basic risk calculations
- Capital efficiency measurements

This is the main testing framework that generates all the detailed results from the whitepaper:

**What it includes:**
- **Monte Carlo Stress Testing** (10,000 simulations)
- **Cross-module synergy analysis**
- **Risk management validation**
- **Capital efficiency calculations**
- **Sensitivity analysis** across key parameters
- **Mathematical model validation**
- **Regulatory compliance checks**

**Expected Output:**
The script will generate:
1. Detailed console output with all test results
2. `sereel_protocol_test_report.txt` - Comprehensive analysis report
3. Interactive visualizations showing:
   - Vault value distributions under stress
   - Health factor analysis
   - Yield breakdowns by module
   - Risk metrics summary

#### Sample Output

When you run `sereel.py`, you'll see results like:

```
SEREEL PROTOCOL COMPREHENSIVE TESTING SUITE
============================================================

1. VAULT INITIALIZATION
   Base Assets: $1,000,000
   AMM Allocation: 40.0%
   Lending Allocation: 40.0%
   Options Allocation: 20.0%

2. EFFECTIVE LIQUIDITY CALCULATION
   Effective Liquidity: $1,653,333
   Capital Efficiency: 1.65x
   Liquidity Increase: 65.3%

3. MODULE YIELD ANALYSIS
   AMM Yield: 2.50%
   Lending Yield: 4.80%
   Options Yield: 8.40%
   Synergy Yield: 1.87%
   Total Yield: 17.57%

...

FINAL PROTOCOL ASSESSMENT:
🚀 Capital Efficiency: 1.65x
💰 Expected Annual Yield: 17.6%
🛡️  99% VaR: 12.3% maximum loss
⚠️  Liquidation Risk: 2.1%
✅ Regulatory Compliance: Verified

🎯 FINAL RECOMMENDATION: RECOMMENDED FOR DEPLOYMENT
```

### Understanding the Results

#### Key Metrics Explained

- **Capital Efficiency**: How many times more effective liquidity the protocol generates compared to simple asset holding
- **Expected Annual Yield**: Projected annual returns from all modules combined
- **99% VaR (Value-at-Risk)**: Maximum expected loss under extreme conditions (99% confidence)
- **Liquidation Risk**: Probability of liquidation events under stress scenarios
- **Health Factor**: Collateral safety margin (>1.0 is safe, <1.0 triggers liquidation)

#### Module Performance

1. **AMM Module**: Automated market making with Uniswap V4 mechanics
2. **Lending Module**: Morpho-style peer-to-peer lending with optimized rates
3. **Options Module**: Black-Scholes options trading adapted for emerging markets
4. **Synergy Effects**: Additional yield from cross-module interactions

### Customizing Parameters

You can modify key parameters in the scripts to test different scenarios:

```python
# In pyramid.py, modify the vault initialization:
vault = SereelVault(
    base_assets=1_000_000,     # Total vault size
    amm_allocation=0.4,        # 40% to AMM
    lending_allocation=0.4,    # 40% to lending  
    options_allocation=0.2     # 20% to options
)
```

### Reproducing Whitepaper Figures

The test suite generates all the charts and data used in the whitepaper sections:

- **Section 4.2**: Capital efficiency calculations
- **Section 5.1**: Module yield analysis
- **Section 6.3**: Risk management metrics
- **Section 7.1**: Monte Carlo stress test results
- **Appendix B**: Mathematical model validations

## Technical Architecture

The implementation follows the exact specifications from the whitepaper:

### Vault Architecture
- Multi-module design with cross-collateralization
- Dynamic rebalancing algorithms
- Liquidity optimization strategies

### Risk Management
- Real-time health factor monitoring
- Automated liquidation mechanisms
- Stress testing framework

### Mathematical Models
- Black-Scholes options pricing with volatility adjustments
- Kinked interest rate curves for lending
- Constant product AMM with impermanent loss calculations

## Contributing

This repository serves as the reference implementation for the Sereel Protocol whitepaper. For questions or discussions about the protocol design, please refer to the whitepaper document.

## Regulatory Compliance

The testing framework includes compliance checks for:
- Capital adequacy requirements
- Position sizing limits
- Risk exposure thresholds
- Institutional investor regulations

## Disclaimer

This repository contains research and testing code for academic and development purposes. The protocols described are experimental and should be thoroughly audited before any production deployment.

## License

This work is released under academic license for research and educational purposes.

---

**For detailed technical specifications, mathematical proofs, and economic analysis, please refer to the complete whitepaper: `surreal.pdf`**
