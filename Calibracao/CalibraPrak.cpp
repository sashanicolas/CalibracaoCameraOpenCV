/*
 * Implementecao Inicial de calibracao de camera com padrao circular
 * Sasha Nicolas
 * Tecgraf PUC Rio
 * 
 * READ.ME:
   1) Passa as imagens em carregaImagens() - hard coded mesmo
   2) Passa a configuracao dos padroes 
      quantPontosControleWidth = 9;
	  quantPontosControleHeight = 6;
	  distanceCP = 35; //milimetros
   3) Configura os #define (ativa ou desativa)
      SELECT_CORNERS_MOUSE - para selecionar com o mouse os 4 cantos de cada imagem
	  MOSTRA_GRID - para visualizar o grid calculado
	  MOSTRA_CADA_ROI - para visulizar cada circulo segmentado (nao ativa - eh meio chato)
	  MOSTRA_POSICAO_PC - para mostra o centro dos pontos de controles (PC)
      MOSTRA_REPROJECAO - mostra comparacao do da reprojecao

	** tudo tem waitkey(0) - entao passa com evento do teclado
	** para selecionar os 4 cantos comeca do canto superior esquerdo e vai girando em ordem horaria
 */

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <math.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

//DEBUG e CONTROLE
#define SELECT_CORNERS_MOUSE 0
#define MOSTRA_GRID 1
#define MOSTRA_CADA_ROI 0
#define MOSTRA_POSICAO_PC 1
#define MOSTRA_REPROJECAO 1

// variaveis
vector<string> imagePaths;
vector<Mat> originalImages;
vector<vector<Point2d>> pontosDoCanto;
int quantPontosControleWidth, quantPontosControleHeight, countPoints, distanceCP;
vector<vector<Point2f>> pontosDoGridDeCadaImagem;
vector<vector<Point2f>> posicaoPontosDeControleEmCadaImagem;
vector<vector<Point3f>> posicaoPontosDeControleIdealObjeto;


// prototipos
void carregaImagens();
void onMouseCallBack(int e, int x, int y, int flags, void * param);
void desenhaCruz(Mat aux, int x, int y, Scalar color);
void desenhaLinha(Mat aux, Point2d a, Point2d b, Scalar color);
void imprimeMat3xn(Mat m, int s);
void calculaGridROI(Mat img, vector<Point2d> pontos, int id);
void mostraImagem(Mat img, string name);
void computeEllipse(Mat img, int idImg);
void cantosPredefinidos();
void selecionarCantosComMouse();
void desenhaQuadrilatero(Mat img, Point p[4], Scalar c);
void calcPosicoesIdeaisObjeto();
static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors);

int main(){
	//configurar definicoes
	quantPontosControleWidth = 9;
	quantPontosControleHeight = 6;
	distanceCP = 35; //milimetros

	// Carregar Imagens
	carregaImagens();

	//Selecionar cantos
	if (SELECT_CORNERS_MOUSE)
		selecionarCantosComMouse();
	else
		cantosPredefinidos();
	
	
	Mat aux;
	
	// Primeira iteracao
	//calcula ponto inicial dos circulos
	for (int i = 0; i < imagePaths.size(); i++){
		aux = originalImages[i].clone();
		computeEllipse(aux, i);
	}

	if (MOSTRA_POSICAO_PC){
		for (int k = 0; k < originalImages.size(); k++){
			aux = originalImages[k].clone();
			for (int i = 0; i < quantPontosControleHeight; i++){
				for (int j = 0; j < quantPontosControleWidth; j++){
					desenhaCruz(aux, posicaoPontosDeControleEmCadaImagem[k][i*quantPontosControleWidth + j].x,
						posicaoPontosDeControleEmCadaImagem[k][i*quantPontosControleWidth + j].y, Scalar(255, 255, 255));
				}				
			}
			mostraImagem(aux, imagePaths[k]);
			moveWindow(imagePaths[k], 150, 150);
			waitKey(0);
			destroyWindow(imagePaths[k]);			
		}
	}

	//calibracao inicial
	//Find intrinsic and extrinsic camera parameters
	Mat cameraMatrix, distCoeffs;
	int flag = CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_ASPECT_RATIO |
		CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_ZERO_TANGENT_DIST;
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flag)
		cameraMatrix.at<double>(0, 0) = 1.0;

	distCoeffs = Mat::zeros(8, 1, CV_64F);
	Size imageSize(originalImages[0].size());
	vector<Mat> rvecs, tvecs;	
	
	calcPosicoesIdeaisObjeto();
	cout << "posicao objeto: "<<posicaoPontosDeControleIdealObjeto[0]<<endl;
	posicaoPontosDeControleIdealObjeto.resize(originalImages.size(), posicaoPontosDeControleIdealObjeto[0]);

	double rms = calibrateCamera(posicaoPontosDeControleIdealObjeto, posicaoPontosDeControleEmCadaImagem, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flag);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;
	
	vector<float> reprojErrs;
	double totalAvgErr = 0;
	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
	totalAvgErr = computeReprojectionErrors(posicaoPontosDeControleIdealObjeto, posicaoPontosDeControleEmCadaImagem,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg re projection error = " << totalAvgErr;
	
	
	
	//outras iteracoes
	//a fazer

	cout << "\nTerminado.";
	int a;
	cin >> a;

	return 0;
}

void carregaImagens(){
	printf("Carregando Imagens\n");
	imagePaths.push_back("imagens/circle1.jpg");
	imagePaths.push_back("imagens/circle2.jpg");
	imagePaths.push_back("imagens/circle3.jpg");
	imagePaths.push_back("imagens/circle4.jpg");
	imagePaths.push_back("imagens/circle5.jpg");

	for (int i = 0; i < imagePaths.size(); i++){
		printf(" - Imagem %d...\n",i);
		Mat aux, aux2;
		aux = imread(imagePaths[i]);
		resize(aux, aux, Size(aux.size().width / 4, aux.size().height / 4));
		originalImages.push_back(aux);
	}
	printf("Feito!\n");	
}

void cantosPredefinidos(){
	Mat aux;
	vector<Point2d> v;

	v.push_back(Point2d(234, 156));
	v.push_back(Point2d(646, 120));
	v.push_back(Point2d(665, 454));
	v.push_back(Point2d(220, 454));
	pontosDoCanto.push_back(v);
	aux = originalImages[0].clone();
	calculaGridROI(aux, pontosDoCanto[0], 0);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[0]);
		waitKey(0);
		cv::destroyWindow(imagePaths[0]);
	}

	v.clear();
	v.push_back(Point2d(184, 147));
	v.push_back(Point2d(594, 161));
	v.push_back(Point2d(642, 455));
	v.push_back(Point2d(202, 507));
	pontosDoCanto.push_back(v);
	aux = originalImages[1].clone();
	calculaGridROI(aux, pontosDoCanto[1], 1);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[1]);
		waitKey(0);
		cv::destroyWindow(imagePaths[1]);
	}

	v.clear();
	v.push_back(Point2d(177, 173));
	v.push_back(Point2d(592, 162));
	v.push_back(Point2d(608, 458));
	v.push_back(Point2d(196, 506));
	pontosDoCanto.push_back(v);
	aux = originalImages[2].clone();
	calculaGridROI(aux, pontosDoCanto[2], 2);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[2]);
		waitKey(0);
		cv::destroyWindow(imagePaths[2]);
	}

	v.clear();
	v.push_back(Point2d(184, 162));
	v.push_back(Point2d(595, 116));
	v.push_back(Point2d(610, 455));
	v.push_back(Point2d(205, 450));
	pontosDoCanto.push_back(v);
	aux = originalImages[3].clone();
	calculaGridROI(aux, pontosDoCanto[3], 3);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[3]);
		waitKey(0);
		cv::destroyWindow(imagePaths[3]);
	}

	v.clear();
	v.push_back(Point2d(149, 134));
	v.push_back(Point2d(644, 96));
	v.push_back(Point2d(718, 483));
	v.push_back(Point2d(124, 512));
	pontosDoCanto.push_back(v);
	aux = originalImages[4].clone();
	calculaGridROI(aux, pontosDoCanto[4], 4);
	if (MOSTRA_GRID) {
		mostraImagem(aux, imagePaths[4]);
		waitKey(0);
		cv::destroyWindow(imagePaths[4]);
	}
}

void selecionarCantosComMouse(){
	// Selecionar 4 pontos limite do padrao para cada foto
	Mat aux;
	for (int i = 0; i < imagePaths.size(); i++){
		countPoints = 0;
		mostraImagem(originalImages[i], imagePaths[i]);
		setMouseCallback(imagePaths[i], onMouseCallBack, &i);

		while (countPoints < 4){
			waitKey(20);
		}

		setMouseCallback(imagePaths[i], NULL, NULL);

		//calcula o grid	
		aux = originalImages[i].clone();
		calculaGridROI(aux, pontosDoCanto[i], i);
		mostraImagem(aux, imagePaths[i]);
		waitKey(0);

		cv::destroyWindow(imagePaths[i]);
	}
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
			desenhaCruz(aux, x, y, Scalar(0, 0, 255));
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

void calculaGridROI(Mat img, vector<Point2d> pontosCanto, int id){
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

	if (MOSTRA_GRID)
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
			y = (float)j / (float)quantPontosControleHeight;
			aux.push_back(Point2f(x, y));
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
		pontos2dImagem[j] = Point2f(pontosGridImagem.at<double>(0, j) / pontosGridImagem.at<double>(2, j),
			pontosGridImagem.at<double>(1, j) / pontosGridImagem.at<double>(2, j));
		// coloca cruzes nos pontos mapeados
		//desenhaCruz(img, pontos2dImagem[j].x, pontos2dImagem[j].y, Scalar(255, 255, 0));
	}

	// Imprime grid
	if (MOSTRA_GRID){
		//linhas horizontais
		x = (quantPontosControleHeight + 1)*(quantPontosControleWidth);
		for (int i = 0; i < quantPontosControleHeight + 1; i++){
			desenhaLinha(img, pontos2dImagem[i],
				pontos2dImagem[i + x], Scalar(255, 0, 0));
		}
		//linhas verticais
		for (int i = 0; i < quantPontosControleWidth + 1; i++){
			desenhaLinha(img, pontos2dImagem[i*(quantPontosControleHeight + 1)],
				pontos2dImagem[i*(quantPontosControleHeight + 1) + quantPontosControleHeight], Scalar(255, 0, 0));
		}
	}

	pontosDoGridDeCadaImagem.push_back(pontos2dImagem);
}

void mostraImagem(Mat img, string name){
	namedWindow(name);
	imshow(name, img);
	moveWindow(name, 150, 150);
}

int _ind(int i, int j){
	return j*(quantPontosControleHeight + 1) + i;
}

Point2d getMenorPonto(Point2f p[4]){
	Point2f c(999999999, 999999999);
	for (int i = 0; i < 4; i++){
		if (p[i].x < c.x) c.x = p[i].x;
		if (p[i].y < c.y) c.y = p[i].y;
	}
	return c;
}

Point2d getMaiorPonto(Point2f p[4]){
	Point2f c(0, 0);
	for (int i = 0; i < 4; i++){
		if (p[i].x > c.x) c.x = p[i].x;
		if (p[i].y > c.y) c.y = p[i].y;
	}
	return c;
}

void desenhaRetangulo(Mat img, Point p1, Point p2, Scalar c){
	rectangle(img, p1, p2, c);
}

void desenhaQuadrilatero(Mat img, Point p[4], Scalar c){
	line(img, p[0], p[1], c);
	line(img, p[1], p[2], c);
	line(img, p[2], p[3], c);
	line(img, p[3], p[0], c);
}

void computeEllipse(Mat imgOriginal, int idImg){
	//Arredonda os pontos do grid de float para inteiro
	vector<Point2f> gridFloat = pontosDoGridDeCadaImagem[idImg];
	vector<Point2d> gridInt(gridFloat.size());
	for (int i = 0; i < gridFloat.size(); i++){
		gridInt[i].x = round(gridFloat[i].x);
		gridInt[i].y = round(gridFloat[i].y);
	}
	
	Mat imgDoLoop;
	double min, max, cweight, thresh;
	int countPixelsInCircle, sumX, sumY;
	Point2f centro;
	vector<Point2f> posicaoPontosDeControle;

	//extrair cada regiao do grid e calcular centro da elipse
	for (int i = 0; i < quantPontosControleHeight; i++){
		for (int j = 0; j < quantPontosControleWidth; j++){
			Point2f p[4];
			p[0] = gridFloat[_ind(i, j)];
			p[1] = gridFloat[_ind(i, j + 1)];
			p[2] = gridFloat[_ind(i + 1, j + 1)];
			p[3] = gridFloat[_ind(i + 1, j)];

			imgDoLoop = imgOriginal.clone();

			Point2f c1 = getMenorPonto(p);
			desenhaCruz(imgDoLoop, c1.x, c1.y, Scalar(0, 0, 255));
			Point2f c2 = getMaiorPonto(p);
			desenhaCruz(imgDoLoop, c2.x, c2.y, Scalar(255, 0, 0));
			int w = c2.x - c1.x, h = c2.y - c1.y;

			Mat cellImg(imgOriginal, Rect(c1.x, c1.y, w, h));

			//threshold it
			Mat cellImgBW(cellImg.rows, cellImg.rows, CV_8UC1);
			cvtColor(cellImg, cellImgBW, CV_RGB2GRAY);
			minMaxLoc(cellImgBW, &min, &max);
			cweight = 0.5;
			thresh = min*cweight + max*(1 - cweight);
			//calcular a media da posicao dos pontos no circulo e fazer o threshold
			countPixelsInCircle = sumX = sumY = 0;
			for (int k = 0; k < cellImgBW.cols; k++){
				for (int l = 0; l < cellImgBW.rows; l++){
					if (cellImgBW.at<uchar>(l, k)>thresh){// na imagem o circulo eh preto
						cellImgBW.at<uchar>(l, k) = 255;						
					}
					else{
						cellImgBW.at<uchar>(l, k) = 0;
						countPixelsInCircle++;
						sumX += k;
						sumY += l;
					}
				}
			}		
			centro.x = ((float)sumX / (float)countPixelsInCircle);
			centro.y = ((float)sumY / (float)countPixelsInCircle);

			desenhaCruz(imgDoLoop, centro.x + c1.x, centro.y + c1.y, Scalar(255, 255, 255));
			desenhaRetangulo(imgDoLoop, c1, c2, Scalar(255, 255, 0));
			//desenhaQuadrilatero(imgDoLoop, p, Scalar(0, 0, 255));
			
			if (MOSTRA_CADA_ROI){
				mostraImagem(imgDoLoop, "ROI");
				mostraImagem(cellImgBW, "Celula");
				moveWindow("Celula", 1000, 150);
				waitKey(0);
			}

			posicaoPontosDeControle.push_back(centro+c1);

		}//for
	}//for

	posicaoPontosDeControleEmCadaImagem.push_back(posicaoPontosDeControle);

	cv::destroyWindow("Celula");
	cv::destroyWindow("ROI");

}

void calcPosicoesIdeaisObjeto(){
	vector<Point3f> v;
	for (int i = 0; i < quantPontosControleHeight; ++i)
	for (int j = 0; j < quantPontosControleWidth; ++j)
		v.push_back(Point3f(float(i*distanceCP), float(j*distanceCP), 0));
	
	posicaoPontosDeControleIdealObjeto.push_back(v);
}

static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	Mat img;
	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		
		if (MOSTRA_REPROJECAO){
			img = originalImages[i].clone();
			
			for (int k = 0; k < quantPontosControleHeight; k++){
				for (int j = 0; j < quantPontosControleWidth; j++){
					desenhaCruz(img, imagePoints[i][k*quantPontosControleWidth + j].x,
						imagePoints[i][k*quantPontosControleWidth + j].y, Scalar(255, 255, 255));
					desenhaCruz(img, imagePoints2[k*quantPontosControleWidth + j].x, 
						imagePoints2[k*quantPontosControleWidth + j].y, Scalar(0, 0, 255));					
				}
			}
			mostraImagem(img, "Reprojecao");
			moveWindow("Reprojecao", 150, 150);
			waitKey(0);
			
		}
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}
	cv::destroyWindow("Reprojecao");
	return std::sqrt(totalErr / totalPoints);
}