#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

fstream out;


vector <Point> anti_cw(vector<Point> cell){
  vector <Point> res;
  for(int i=cell.size(); i>1; i--){
    res.push_back(cell[i]);
  }
  res.insert(res.begin(),cell[0]);
  return res;
}

Mat LBP(Mat img, int nneighbors, int nspaces, int cw){
  int rows = img.size().height;
  int cols = img.size().width;
  Mat ILBP = Mat::zeros( img.size(), CV_8UC1 );

  for(int x=0; x<cols; x++){
    for(int y=0; y<rows; y++){
      Point p = Point(x,y);
      vector<Point> cell;
      int nx,ny;
      int itr=(8*nspaces)/nneighbors;
      for(int i=-nspaces; i<=nspaces; i+=itr){
        nx=x+i;
        ny=y-nspaces;
        cell.push_back(Point(nx,ny));
      }
      for(int i=-(nspaces-itr); i<=nspaces; i+=itr){
        nx=x+nspaces;
        ny=y+i;
        cell.push_back(Point(nx,ny));
      }
      for(int i=nspaces-itr; i>=-nspaces; i-=itr){
        nx=x+i;
        ny=y+nspaces;
        cell.push_back(Point(nx,ny));
      }
      for(int i=nspaces-itr; i>=-(nspaces-1); i-=itr){
        nx = x-nspaces;
        ny = y+i;
        cell.push_back(Point(nx,ny));
      }
      if(cw==0)
        cell = anti_cw(cell);
      int val_LBP=0;
      for(int i=0; i<cell.size(); i++){
        if(cell[i].y > 0 && cell[i].y<rows && cell[i].x > 0 && cell[i].x<cols){
          Scalar intensity = img.at<uchar>(cell[i].y, cell[i].x);
          float v = intensity.val[0];
          intensity = img.at<uchar>(y, x);
          float vp = intensity.val[0];
          if(v>=vp)
            val_LBP += pow(2,i);
        }
      }
      ILBP.at<uchar>(Point(x,y)) = val_LBP;
    }
  }
  int histSize = 256;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true; bool accumulate = false;
  Mat hist;
  calcHist( &ILBP, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
  normalize(hist, hist, 0.0, 1.0, NORM_MINMAX, CV_32FC1);

  Mat code = Mat::zeros( hist.size(), CV_32FC2 );
  for(int i=0; i<hist.size().height;i++){
    Vec2f aux;
    aux[0] = i;
    aux[1] =  hist.at<float>(i);
    code.at<Vec2f>(i) = aux;
  }
  return code;
}

Mat ULBP(Mat img, int nneighbors, int nspaces, int cw){
  Mat hist = LBP(img, nneighbors, nspaces, cw);
  int uniform[] = {
    0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255
  };
  int uTam = 58;
  Mat code = Mat::zeros( Size(1,uTam), CV_32FC2 );
  for(int i=0; i<uTam;i++){
    code.at<Vec2f>(i) = hist.at<Vec2f>(uniform[i]);
  }
  return code;
}

void writeCSV(Mat ULBP, int val){
  out << val << ",";
  for(int i=0; i<ULBP.size().height-1;i++){
    Vec2f intensity = ULBP.at<Vec2f>(i);
    if(i+1!=ULBP.size().height-1)
      out << float(intensity[1]) << " ";
    else
      out << float(intensity[1]);
  }
  out << endl;
}

int main(int argc, char** argv){
  out.open("vec.csv");
  out << "out,vec" << endl;
  if(argc < 3){
    cout << "./main pos.txt neg.txt" << endl;
    return -1;
  }
  fstream pos;
  pos.open(argv[1]);
  char line[256];
  while(pos.getline(line,256)){
    Mat img;
    img = imread(line, CV_LOAD_IMAGE_GRAYSCALE );;
    Mat imgULBP = ULBP(img,8,1,1);
    writeCSV(imgULBP,1);
  }
  pos.close();
  fstream neg;
  neg.open(argv[2]);
  while(neg.getline(line,256)){
    Mat img;
    img = imread(line, CV_LOAD_IMAGE_GRAYSCALE );;
    Mat imgULBP = ULBP(img,8,1,1);
    writeCSV(imgULBP,0);
  }
  out.close();
  return 0;
}
