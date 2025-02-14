struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};
    double sum_0 = 0;
    double sum_1 = 0;
    double sum_2 = 0;
    for(int row=x0;row<x1;row++){
        for(int col=y0;col<y1;col++){
            sum_0 += data[3 * row + 3 * nx * col];
            sum_1 += data[1 + 3 * row + 3 * nx * col];
            sum_2 += data[2 + 3 * row + 3 * nx * col];
        }
    }
        
        result.avg[0] = sum_0 / ((x1-x0)*(y1-y0));
        result.avg[1] = sum_1 / ((x1-x0)*(y1-y0));
        result.avg[2] = sum_2 / ((x1-x0)*(y1-y0));
    return result;
}
