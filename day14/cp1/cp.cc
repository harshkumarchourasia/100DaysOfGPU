#include <cmath>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

void correlate(int ny, int nx, const float *data, float *result)
{
    double *mean;
    mean = new double[ny];
    for (int j = 0; j < ny; j++)
    {
        mean[j] = 0;
        double temp = 0;
        for (int i = 0; i < nx; i++)
        {
            temp += data[i + j * nx];
        }
        mean[j] = temp / nx;
    }

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < ny; j++)
        {   
            if (j > i)
            {
                result[i + j * ny] = 0;
                continue;
            }
            double sum_0 = 0;
            double sum_1 = 0;
            double sum_2 = 0;
            for (int idx = 0; idx < nx; idx++)
            {
                sum_0 += (data[idx + i * nx] - mean[i]) * (data[idx + j * nx] - mean[j]);
                sum_1 += (data[idx + i * nx] - mean[i]) * (data[idx + i * nx] - mean[i]);
                sum_2 += (data[idx + j * nx] - mean[j]) * (data[idx + j * nx] - mean[j]);
            }
            result[i + j * ny] = sum_0 / sqrt(sum_1 * sum_2);
        }
    }
    delete[] mean;
}