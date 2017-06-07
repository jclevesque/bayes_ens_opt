#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void majority_vote(long* votes, long* unique_labels, long* predictions,
  int n, int m, int nl)
{
    int i,v;
    int row, col;
    int cur_vote, max_votes;
    int prediction;
    long* votes_by_label;

    votes_by_label = (long*) malloc((n * nl) * sizeof(long));
    if (votes_by_label == NULL)
    {
        printf("Error allocating memory for votes. \n");
        exit(1);
    }

    for (i = 0; i < n*nl; i++)
        votes_by_label[i] = 0;

    //Iterate over all samples
    for (row = 0; row < n; row++)
    {
        //First count votes.
        for (col = 0; col < m; col++)
        {
            cur_vote = votes[row*m + col];
            for (v = 0; v < nl; v++)
            {
                if(cur_vote == unique_labels[v])
                    votes_by_label[row*nl + v] += 1;
            }
        }

        //Then find label with maximum number of votes.
        max_votes = -1;
        prediction = 0;
        for (v=nl - 1; v >= 0; v--)
        {
            if(votes_by_label[row*nl + v] >= max_votes)
            {
                max_votes = votes_by_label[row*nl + v];
                prediction = unique_labels[v];
            }
        }
        predictions[row] = prediction;
    }

    free(votes_by_label);
}

void weighted_vote(long* votes, double* weights, long* unique_labels, long* predictions,
  int n, int m, int nl)
{
    int i,v;
    int row, col;
    int cur_vote;
    double max_score;
    int prediction;
    double* score_by_label;

    //I don't why but this helps.
    //nl = nl;
    score_by_label = (double*) malloc(n * nl * sizeof(double));
    if (score_by_label == NULL)
    {
        printf("Error allocating memory for scores. (Tried to get %ld bytes)\n",
         n * nl * sizeof(double));
        exit(1);
    }

    for (i = 0; i < n*nl; i++)
        score_by_label[i] = 0;

    //Iterate over all samples
    for (row = 0; row < n; row++)
    {
        //First count votes.
        for (col = 0; col < m; col++)
        {
            cur_vote = votes[row*m + col];
            for (v = 0; v < nl; v++)
            {
                if(cur_vote == unique_labels[v])
                    score_by_label[row*nl + v] += weights[col];
            }
        }

        //Then find label with maximum number of votes.
        max_score = 0.;
        prediction = 0;
        for (v=nl - 1; v >= 0; v--)
        {
            if(score_by_label[row*nl + v] >= max_score)
            {
                max_score = score_by_label[row*nl + v];
                prediction = unique_labels[v];
            }
        }
        predictions[row] = prediction;
    }

    free(score_by_label);
}


double diversity_kw(long* votes, long* labels, int n, int m)
{
    int i,j;
    double kw;
    double correct;

    /* Python code:
    sum = 0
    for i in range(n):
        correct = np.sum(votes[i] == labels[i])
        sum += correct * (m - correct)
    sum /= m

    kw = sum / (n * m**2)*/

    kw = 0;
    for (i = 0; i < n; i++)
    {
        correct = 0;
        for (j = 0; j < m; j++)
        {
            if (labels[i] == votes[i*m + j])
                correct += 1;
        }
        kw += correct * (m - correct);
    }
    kw = kw / (n * m * m);
    return kw;
}

double diversity_entropy(long* votes, long* labels, int n, int m)
{
    int i,j;
    double H;
    double correct, a, b;

    /* Python code:
    n, m = votes.shape
    E = 0
    for i in range(n):
        c = np.sum(votes[i] == labels[i]) / m
        a = c + 0.01
        inv_a = (1 - c) + 0.01
        E += -a*np.log(a) - (inv_a)*np.log(inv_a)
    E = 2 * E / n */

    H = 0;
    for (i = 0; i < n; i++)
    {
        correct = 0;
        for (j = 0; j < m; j++)
        {
            if (labels[i] == votes[i*m + j])
                correct += 1;
        }
        correct /= m;
        a = correct + 0.01;
        b = (1 - correct) + 0.01;
        H += -a * log(a) - (b) * log(b);
    }
    H = 2 * H / n;
    return H;
}