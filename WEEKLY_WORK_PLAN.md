Weekly Work Plan: Neural Architecture Search with Approximate Multipliers

OBJECTIVES

This week focuses on aligning the implementation with the approxAI paper architecture while debugging current issues and adding visualization capabilities. The main goals are investigating low accuracy (around 70 percent instead of expected 91-92 percent), understanding STL constraint violations, implementing missing plotting functionality, relaxing STL constraints for realistic evaluation, enhancing Bayesian NAS, and gaining overview of differential privacy concepts. The architecture implementation will follow the paper's approach as discussed in meetings.


CURRENT STATUS

The implementation shows 71 percent STL constraint violation rate. Best accuracy achieved is around 70 percent on CIFAR-10. The code currently outputs results only to terminal with no plots or visualizations. Energy calculations use incomplete multiplier data. Bayesian optimization is being added and needs testing. Current architecture needs alignment with paper specifications.


WEEKLY SCHEDULE


Monday, November 18

Investigate the low accuracy problem by comparing current architecture with paper specifications. Review approxAI paper to verify ResNet configuration matches their implementation. Run baseline ResNet-20 experiment following paper's exact architecture to establish expected accuracy. Check if data augmentation and learning rate schedule match paper's training procedure.

Analyze STL constraint violations in detail following paper's methodology. Observe which architectures violate quality threshold versus energy threshold. Calculate violation percentages. Review whether Qc=0.80 and Ec=100.0 mJ align with paper's constraint values.

Meeting scheduled at 12:00 PM with Amjad (TA) to discuss Verilog and VHDL for the next semester coursework. Take notes on digital design concepts.

Review meeting notes from Amjad session. Organize key points about hardware design that relate to understanding multiplier implementations as described in the paper.


Tuesday, November 19

Verify architecture implementation matches paper specifications. Check that ResNet stages, blocks per stage, and filter configurations align with paper's ResNet-18/34 architectures. Adjust implementation if discrepancies are found.

Implement plotting and logging functionality for training curves following paper's result presentation style. Add code to save and plot accuracy over epochs during training. Plot validation accuracy and loss curves. Save plots to files automatically after each trial.

Implement accuracy versus energy scatter plots as shown in the paper. Create function to plot all evaluated architectures. Highlight Pareto-optimal solutions similar to paper's figures.

Create Pareto front visualization matching paper's style. Plot non-dominated solutions. Include STL satisfaction indicators as described in paper methodology.


Wednesday, November 20

Relax STL constraints to match paper's approach more closely. Review paper's quality and energy constraints. Adjust Qc and Ec based on paper's values and Monday's empirical analysis. Run experiments with constraints aligned to paper methodology.

Implement visualization for STL robustness scores following paper's evaluation framework. Plot robustness values across all trials. Show which trials satisfy constraints per paper's criteria.

Meeting scheduled at 12:00 PM with Amjad (TA) for continued discussion on digital design. Take notes on hardware synthesis and timing concepts.

Process meeting notes and understand the tools and concepts required.


Thursday, November 21

Investigate Bayesian optimization implementation ensuring it follows optimization principles similar to paper's NSGA-II approach. Run small experiments with 10 trials to verify functionality. Compare search efficiency to paper's reported results.

Implement Bayesian optimization visualization. Observe and note down the convergence to compare with paper's search convergence patterns.

Create comprehensive logging system to track experiments similar to paper's methodology. Replace print statements with logging module. Save detailed experiment logs.

Read overview materials on differential privacy concepts. Understand basic principles. Note down potential future work ideas.


Friday, November 22

Generate binary multiplier files with energy data using specifications from EvoApproxLib referenced in the paper. Extract power consumption values for mul8u_197B, mul8u_1JJQ, and mul8u_0AB. Create binary format matching paper's energy model requirements.

Update energy_calculator.py to match paper's energy calculation methodology. Verify calculations produce values in same range as paper's reported energy consumption. Test with architectures used in the paper.

Run controlled baseline experiment using paper's exact ResNet configurations. Train without approximation to establish baseline matching paper's exact model accuracy. Note down results for comparison.

Continue reading about differential privacy. Focus on understanding privacy-preserving approaches that could complement approximate computing research direction.


EXPECTED OUTCOMES

By end of the work week (Friday), architecture should closely match paper specifications. Plotting functionality should produce visualizations comparable to paper's figures. STL constraints should align with paper's methodology. Accuracy should approach paper's reported 91-92 percent for ResNet. Energy calculations should use same model as paper. Bayesian NAS should be validated. Comprehensive logs and visualizations following paper's style should be available. Basic understanding of differential privacy concepts should be gained.


RESOURCE REQUIREMENTS

Software needed includes matplotlib, seaborn, pandas, TensorFlow, and scikit-learn.

Computational resources include Singularity machine for GPU experiments and MacBook for development.

Reference materials include approxAI paper as primary guide, EvoApproxLib specifications referenced in paper, and differential privacy introductory materials.


NOTES

The implementation follows the approxAI paper architecture and methodology as discussed in advisor meetings. The two meetings with Amjad on Monday and Wednesday focus on Verilog and VHDL for next semester and are separate from the 4-hour research commitment.

Progress will be tracked daily with notes on alignment with paper specifications. Code will be committed to git regularly.
