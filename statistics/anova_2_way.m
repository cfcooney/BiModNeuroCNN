% Ciaran Cooney, 2020
% Script for performing 2-way ANOVA with post-hoc analysis 
% using the Tukey Honest Significant Difference criterion.
%
% Data Structure:
%                                  Column Variables
%      Row Variables          Condition 1 | Condition 2
% Condition 1, Replication 1      10      |      13
% Condition 1, Replication 2      12      |      14
% Condition 2, Replication 1       5      |       5
% Condition 2, Replication 2       7      |       9
% Condition 3, Replication 1      18      |      16
% Condition 3, Replication 2      11      |      10

clear all
path = '/';

y = xlsread([path 'testing_data.xlsx'],'overt', 'B2:D57');

replications = 28; % number of sessions

[p,tbl,stats] = anova2(y, replications);
tbl

% Pairwise comparison of the column data
[c, m, h, nms] = multcompare(stats,'alpha',.05,'ctype','hsd'); %p-values returned in 'c' variable.

figure
% Pairwise comparison of the row data
[cR, mR, hR, nmsR] = multcompare(stats,'alpha',.05,'Estimate','row','ctype','hsd'); %p-values returned in 'c' variable.
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear all
path = '/';

y = xlsread([path 'testing_data.xlsx'],'imagined', 'B2:D43');

replications = 21; % number of sessions

[p,tbl,stats] = anova2(y, replications);
tbl

% Pairwise comparison of the column data
[c, m, h, nms] = multcompare(stats,'alpha',.05,'ctype','hsd'); %p-values returned in 'c' variable.

figure
% Pairwise comparison of the row data
[cR, mR, hR, nmsR] = multcompare(stats,'alpha',.05,'Estimate','row','ctype','hsd'); %p-values returned in 'c' variable.
