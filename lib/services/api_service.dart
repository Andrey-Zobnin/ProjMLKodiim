import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {
  static const String _baseUrl = "http://localhost:8080"; // Или IP сервера

  // GET-запрос
  static Future<String> fetchData() async {
    final response = await http.get(Uri.parse('$_baseUrl/api/message'));
    if (response.statusCode == 200) {
      return json.decode(response.body)['text'];
    } else {
      throw Exception('Failed to load data');
    }
  }

  // POST-запрос
  static Future<String> sendData(String text) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/api/data'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': text}),
    );
    if (response.statusCode == 200) {
      return json.decode(response.body)['received'];
    } else {
      throw Exception('Failed to send data');
    }
  }
}