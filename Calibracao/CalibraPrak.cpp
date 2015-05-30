/*
 * Implementecao Inicial de calibracao de camera com padrao circular
 * Sasha Nicolas
 * Tecgraf PUC Rio
 */

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

// variaveis
vector<string> imagePaths;
vector<Mat> originalImages;
vector<vector<Point2d>> pontosDoCanto;
int quantPontosControleWidth, quantPontosControleHeight;
int countPoints;
vector<vector<vector<Point2d>>> pontosDoGridDeCadaImagem;

// prototipos
void carregaImagens();
void onMouseCallBack(int e, int x, int y, int flags, void * param);
void desenhaCruz(Mat aux, int x, int y, Scalar color);
void desenhaGrid(Mat img, vector<Point2d> pontos, int id);
void mostraImagem(Mat img, string name);

int main(){
	//carregar definicoes
	quantPontosControleWidth = 9;
	quantPontosControleHeight = 6;

	// Selecionar Imagens
	carregaImagens();

	// Selecionar 4 pontos limite do padrao para cada foto
	Mat aux;
	for (int i = 0; i < imagePaths.size(); i++){
		countPoints = 0;
		mostraImagem(originalImages[i], imagePaths[i]);
		setMouseCallback(imagePaths[i], onMouseCallBack, &i);

		while (countPoints<4){
			waitKey(20);
		}

		setMouseCallback(imagePaths[i], NULL, NULL);
		
		//desenhar o grid	
		aux = originalImages[i].clone();
		desenhaGrid(aux, pontosDoCanto[i], i);
		mostraImagem(aux, imagePaths[i]);
		waitKey(0);

		destroyWindow(imagePaths[i]);		
	}

	return 0;
}

void carregaImagens(){
	printf("Carregando Imagens\n");
	imagePaths.push_back("imagens/circle1.jpg");
	imagePaths.push_back("imagens/circle2.jpg");
	/*imagePaths.push_back("imagens/circle3.jpg");
	imagePaths.push_back("imagens/circle4.jpg");
	imagePaths.push_back("imagens/circle5.jpg");*/

	for (int i = 0; i < imagePaths.size(); i++){
		Mat aux, aux2;
		aux = imread(imagePaths[i]);
		resize(aux, aux, Size(aux.size().width / 4, aux.size().height / 4));
		originalImages.push_back(aux);
	}
	printf("Feito!\n");
}

void onMouseCallBack(int e, int x, int y, int flags, void * imageId){
	if (e == CV_EVENT_LBUTTONDOWN){
		int id = *((int*)imageId);
		printf("x,y (%d,%d)\n", x, y);

		Point2d ponto(x, y);

		if (id >= pontosDoCanto.size()){
			vector<Point2d> v;
			pontosDoCanto.push_back(v);
		}
		pontosDoCanto[id].push_back(ponto);

		// colocar marca no ponto clicado
		Mat aux = originalImages[id].clone();
		for (int i = 0; i < pontosDoCanto[id].size(); i++){
			int x = pontosDoCanto[id][i].x, y = pontosDoCanto[id][i].y;
			desenhaCruz(aux, x, y, Scalar(0,0,255));
		}

		mostraImagem(aux, imagePaths[id]);

		countPoints++;
	}
}

void desenhaCruz(Mat aux, int x, int y, Scalar color){
	line(aux, Point(x - 5, y), Point(x + 5, y), color, 1, 8);
	line(aux, Point(x, y - 5), Point(x, y + 5), color, 1, 8);
}

void desenhaLinha(Mat aux, Point2d a, Point2d b, Scalar color){
	line(aux, a, b, color, 1, 8);
}

void imprimeMat3xn(Mat m, int s){
	cout << "[" << endl;
	for (int j = 0; j < s; j++){
		cout << " " << m.at<double>(0, j) << ", " << m.at<double>(1, j) << ", " << m.at<double>(2, j) << endl;
	}
	cout << "]" << endl;
}

void desenhaGrid(Mat img, vector<Point2d> pontosCanto, int id){
	vector<Point2f> planePoints, imagePoints;
	
	planePoints.push_back(Point2f(0, 0));
	planePoints.push_back(Point2f(1, 0));
	planePoints.push_back(Point2f(1, 1));
	planePoints.push_back(Point2f(0, 1));

	imagePoints.push_back(Point2f(pontosCanto[0].x, pontosCanto[0].y));
	imagePoints.push_back(Point2f(pontosCanto[1].x, pontosCanto[1].y));
	imagePoints.push_back(Point2f(pontosCanto[2].x, pontosCanto[2].y));
	imagePoints.push_back(Point2f(pontosCanto[3].x, pontosCanto[3].y));

	Mat H = findHomography(planePoints, imagePoints);

	cout << "H = " << endl << " " << H << endl << endl;
		
	int quantPontosGrid = (quantPontosControleWidth + 1) * (quantPontosControleHeight + 1);
	Mat pontosGridPlano(3, quantPontosGrid, CV_64F);
	Mat pontosGridImagem(3, quantPontosGrid, CV_64F);
	vector<Point2f> pontos2dImagem(quantPontosGrid);
	vector<Point2f> aux;

	float x, y;
	
	//calculando pontos no plano
	for (int i = 0; i < quantPontosControleWidth + 1; i++){
		x = (float)i / (float)quantPontosControleWidth;
		for (int j = 0; j < quantPontosControleHeight + 1; j++){
			y = (float)j/(float)quantPontosControleHeight;
			aux.push_back(Point2f(x,y));
		}		
	}
	//cout << "aux = " << endl << " " << aux << endl;
	//passa do vetor para a matriz
	
	for (int j = 0; j < quantPontosGrid; j++){
		pontosGridPlano.at<double>(0, j) = aux[j].x;
		pontosGridPlano.at<double>(1, j) = aux[j].y;
		pontosGridPlano.at<double>(2, j) = 1;	
	}
	
	//multiplica pela homographia	
	pontosGridImagem = H * pontosGridPlano;		

	//passa para o vetor
	for (int j = 0; j < quantPontosGrid; j++){
		//dividir coordenadas homogeneas pelo z
		pontos2dImagem[j] = Point2f(pontosGridImagem.at<double>(0, j) / pontosGridImagem.at<double>(2,j), 
			pontosGridImagem.at<double>(1, j) / pontosGridImagem.at<double>(2, j));
		// coloca cruzes nos pontos mapeados
		//desenhaCruz(img, pontos2dImagem[j].x, pontos2dImagem[j].y, Scalar(255, 255, 0));
	}

	// Imprime grid
	//linhas horizontais
	x = (quantPontosControleHeight + 1)*(quantPontosControleWidth);
	for (int i = 0; i < quantPontosControleHeight + 1; i++){
		desenhaLinha(img, pontos2dImagem[i],
			pontos2dImagem[i+ x], Scalar(255, 0, 0));
	}
	//linhas verticais
	for (int i = 0; i < quantPontosControleWidth + 1; i++){
		desenhaLinha(img, pontos2dImagem[i*(quantPontosControleHeight + 1)],
			pontos2dImagem[i*(quantPontosControleHeight + 1) + quantPontosControleHeight], Scalar(255, 0, 0));
	}

}

void mostraImagem(Mat img, string name){
	namedWindow(name);
	imshow(name, img);
	moveWindow(name, 150, 150);
}