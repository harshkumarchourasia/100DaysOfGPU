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
    for(int color=0; color<3;color++){
        double sum = 0;
        for(int row=x0;row<x1;row++){
            for(int col=y0;col<y1;col++){
                sum += data[color + 3 * row + 3 * nx * col];
            }
        }
        result.avg[color] = sum / ((x1-x0)*(y1-y0));
    }
    return result;
}
